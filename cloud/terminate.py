import requests, os

if __name__ == "__main__":
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("lol get trolled api key not found have fun when you wake up")
        raise Exception("api key not found")

    get_ip = "https://api.ipify.org"

    base_url = "https://cloud.lambda.ai"
    get_instances = f"{base_url}/api/v1/instances"
    terminate_instances = f"{base_url}/api/v1/instance-operations/terminate"

    for i in range(3):
        try:
            public_ip = requests.get(get_ip).text

            running_instances = requests.get(get_instances, auth=(api_key, ""))
            instances = running_instances.json()["data"]

            ids = [instance["id"] for instance in instances if instance["ip"] == public_ip]
            
            if (len(ids) > 0):
                requests.post(terminate_instances, json={"instance_ids": ids}, auth=(api_key, ""))
            else:
                print("Not an active instance.")

        except Exception as e:
            print(e)