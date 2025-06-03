import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

'''
Define a function to get the Azure OpenAI client.
'''
def get_az_openai_client():

    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key=os.getenv("AZURE_OPENAI_API_KEY")

    client = OpenAI(
        api_key=api_key,  
        base_url=f"{azure_endpoint}openai/v1/",
        default_query={"api-version": "preview"}, 
    )
    return client

class MCPClient:
    def __init__(self, provider: str = "azure"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.provider = provider
        
        if self.provider.lower() == "azure":
            self.openai_agent = get_az_openai_client()
        else: 
            self.openai_agent = OpenAI()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get a list of available tools from the server
        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "type": "function",
            "parameters": tool.inputSchema
        } for tool in response.tools]

        # Initial Open AI API call
        response = self.openai_agent.responses.create(
                model="gpt-4o",
                tools=available_tools,
                input=messages
                )

        # Process response and handle tool calls
        final_text = []

        for tool_call in response.output:
            if tool_call.type != "function_call":
                final_text.append(response.output[0].content.text)

            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)

            final_text.append(f"[Calling tool {tool_name} with args {tool_args}] \n")

            result = await self.session.call_tool(tool_name, tool_args)

            final_text.append(f"{str(result)} \n")

            # Append the tool call and result to the messages
            messages.append(tool_call)

            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            })

            # Get final response from LLM based on the result
            response_natural_text = self.openai_agent.responses.create(
                model="gpt-4o",
                tools=available_tools,
                input=messages
                )
            
            final_text.append(response_natural_text.output_text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response + "\n")

            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main(provider: str):
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient(provider=provider)
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main(provider="azure"))