# app/config.py
import os
from dotenv import load_dotenv

def log_step(category: str, message: str) -> None:
    print(f"{category}: {message}")

from typing import Dict, List

def validate_env_vars(var_dict: Dict[str, str]) -> None:
    errors: List[str] = []
    for key, value in var_dict.items():
        if not value:
            errors.append(f"{key} is missing or empty.")
        elif value.strip() != value:
            errors.append(f"{key} has leading/trailing whitespace.")
    if errors:
        for err in errors:
            print(f"[VALIDATION] {err}")
        raise EnvironmentError("Environment variable validation failed.")

from typing import Optional

def load_api_keys(dotenv_file: Optional[str] = None):
    log_step("Config", "Loading API keys from environment or secrets")

    if not dotenv_file:
        dotenv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.local'))
    load_dotenv(dotenv_path=dotenv_file, override=True)
    log_step("Config", f"Loaded dotenv from: {dotenv_file}")

    fmp_api_key = os.getenv("FMP_API_KEY")
    ibm_cloud_api_key = os.getenv("IBM_CLOUD_API_KEY")
    ibmq_api_token = os.getenv("IBMQ_API_TOKEN")

    print("\n[DEBUG] Environment Variables:")
    print("FMP_API_KEY:", repr(fmp_api_key))
    print("IBM_CLOUD_API_KEY:", repr(ibm_cloud_api_key))
    print("IBMQ_API_TOKEN:", repr(ibmq_api_token))

    validate_env_vars({
        key: value for key, value in {
            "FMP_API_KEY": fmp_api_key,
            "IBM_CLOUD_API_KEY": ibm_cloud_api_key,
            "IBMQ_API_TOKEN": ibmq_api_token
        }.items() if value is not None
    })

    if os.getenv("IBM_CLOUD_ENV") == "true":
        log_step("Config", "Detected IBM Cloud environment, loading secrets")
        try:
            with open('/mnt/secrets-store/FMP_API_KEY', 'r') as f:
                fmp_api_key = f.read().strip()
            with open('/mnt/secrets-store/IBM_CLOUD_API_KEY', 'r') as f:
                ibm_cloud_api_key = f.read().strip()
            with open('/mnt/secrets-store/IBMQ_API_TOKEN', 'r') as f:
                ibmq_api_token = f.read().strip()
        except Exception as e:
            log_step("Config", f"Failed to load IBM Cloud secrets: {str(e)}")
            raise EnvironmentError("IBM Cloud secrets missing or inaccessible")

    log_step("Config", "All API keys loaded successfully")
    return fmp_api_key, ibm_cloud_api_key, ibmq_api_token

if __name__ == "__main__":
    try:
        keys = load_api_keys()
        print("\n✅ Keys loaded for test:")
        print("FMP_API_KEY:", keys[0][:6] + "..." if keys[0] else "MISSING")
        print("IBM_CLOUD_API_KEY:", keys[1][:6] + "..." if keys[1] else "MISSING")
        print("IBMQ_API_TOKEN:", keys[2][:6] + "..." if keys[2] else "MISSING")
    except Exception as e:
        print("❌ Error:", e)
