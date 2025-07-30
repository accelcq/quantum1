#!/usr/bin/env python3
"""
IBM Quantum API Token Setup Script

This script helps you set up your IBM Quantum API token for the Quantum1 application.
"""

import os
import sys
from pathlib import Path

def print_header():
    print("=" * 60)
    print("IBM Quantum API Token Setup")
    print("=" * 60)

def print_instructions():
    print("\n📋 Instructions:")
    print("1. Go to https://quantum.ibm.com/")
    print("2. Sign in with your IBM account")
    print("3. Go to 'Account' → 'API token'")
    print("4. Create a new token or copy an existing one")
    print("5. The token should look like: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...")
    print("\n⚠️  Keep your token secure and never share it publicly!")

def validate_token(token):
    """Basic validation of the token format"""
    if not token or len(token.strip()) < 10:
        return False, "Token is too short or empty"
    
    # Check if it looks like a JWT token (starts with eyJ)
    if not token.strip().startswith('eyJ'):
        return False, "Token doesn't appear to be in the correct format"
    
    return True, "Token format looks valid"

def setup_token():
    print_header()
    print_instructions()
    
    # Check if token already exists
    current_token = os.getenv("IBMQ_API_TOKEN")
    if current_token:
        print(f"\n🔍 Found existing token: {current_token[:10]}...")
        response = input("Do you want to replace it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Get new token
    print("\n" + "=" * 40)
    token = input("Enter your IBM Quantum API token: ").strip()
    
    # Validate token
    is_valid, message = validate_token(token)
    if not is_valid:
        print(f"❌ {message}")
        return
    
    # Save token to .env file
    env_file = Path(__file__).parent.parent / ".env.local"
    env_file.parent.mkdir(exist_ok=True)
    
    # Read existing .env file
    env_content = ""
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
    
    # Update or add IBMQ_API_TOKEN
    lines = env_content.split('\n') if env_content else []
    token_line_found = False
    
    for i, line in enumerate(lines):
        if line.startswith('IBMQ_API_TOKEN='):
            lines[i] = f'IBMQ_API_TOKEN={token}'
            token_line_found = True
            break
    
    if not token_line_found:
        lines.append(f'IBMQ_API_TOKEN={token}')
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Token saved to {env_file}")
    print("\n🔧 Next steps:")
    print("1. Restart your application to load the new token")
    print("2. Test the connection using the quantum endpoints")
    print("3. If you're using Kubernetes, update your secrets")

def test_connection():
    """Test the IBM Quantum connection"""
    print("\n🧪 Testing IBM Quantum connection...")
    
    try:
        read_dotenv()
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.getenv("IBMQ_API_TOKEN")
        
        if not token:
            print("❌ No IBMQ_API_TOKEN found in environment")
            return False
        
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backends = service.backends()
        
        print(f"✅ Connection successful! Found {len(backends)} backends:")
        for backend in backends[:5]:  # Show first 5 backends
            print(f"   - {backend.name}")
        
        if len(backends) > 5:
            print(f"   ... and {len(backends) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
    else:
        setup_token() 