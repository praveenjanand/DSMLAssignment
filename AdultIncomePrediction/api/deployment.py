import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_R5ShDT1qbuhv14wifNHJ9NmLJGvtge41Q2Xe"  # Your Prefect Cloud API key
ACCOUNT_ID = "f5fe7ddd-15ca-439f-a5fb-304dbbfc4064"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "7f1bb8d2-0a07-4f1e-abe2-3c271406bb43"  # Your Prefect Cloud Workspace ID
DEPLOYMENT_ID = "bda7420a-c6d0-4f5f-977f-46bc31a3c350"  # Your Deployment ID

# Correct API URL to get deployment details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/deployments/{DEPLOYMENT_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request using GET
response = requests.get(PREFECT_API_URL, headers=headers)

# Check the response status
if response.status_code == 200:
    deployment_info = response.json()
    print(deployment_info)
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")