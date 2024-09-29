# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# URL - https://app.prefect.cloud/api/docs

import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "cli-4fdcf0e3-d4c6-44c9-b493-c2d614d608b4"  # Your Prefect Cloud API key
ACCOUNT_ID = "35053e5f-52e7-416d-a906-a047d8b21ca4"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "04d8fca9-df2e-40c8-ae4f-a3733114c475"  # Your Prefect Cloud Workspace ID
DEPLOYMENT_ID = "0ba4ca1b-8471-4acf-a121-8568ce7b5f51"  # Your Deployment ID

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
