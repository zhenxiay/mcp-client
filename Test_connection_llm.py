
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

def get_alerts(state):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    response = requests.get(f"{NWS_API_BASE}/alerts/active/area/{state}", 
                            headers=headers)
    data = response.json()

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]

    return "\n---\n".join(alerts)

tools = [{
    "type": "function",
    "name": "get_alerts",
    "description": "Get weather alerts for a US state.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {"type": "string"}
        },
        "required": ["state"],
        "additionalProperties": False
    },
    "strict": True
}]

def connection_test(service):
    #Get initial answer (functional call and arguments) from LLM
    if service == "openai":
        client = OpenAI()
        messages = [ {"role": "user", "content": "Are the any weather alerts in Utah?"} ]
        response = client.responses.create(
                                      model="gpt-4.1",
                                      tools=tools,
                                      input=messages
                                       )
    else:
        model = "claude-3-5-sonnet-20241022"
        client = Anthropic()
        messages = [ {"role": "user", "content": "Hello, Claude"} ]
        response = client.messages.create(
              model=model,
              max_tokens=1000,
              messages=messages
             )

    # Process initial response and pass arguments for tool call    
    tool_call = response.output[0]
    args = json.loads(tool_call.arguments)
    result = get_alerts(args["state"])

    messages.append(tool_call)  
    messages.append({                               
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

    # Get final response from LLM
    response_natural_text = client.responses.create(
        model="gpt-4.1",
        input=messages,
        tools=tools,
    )

    return response, response_natural_text

if __name__ == "__main__":
    # input "openai" or "anthropic" as service
    response, response_natural_text = connection_test(service="openai")   

    print(response.output)
    print(response.output[0].type)
    print(response_natural_text.output_text)