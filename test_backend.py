#!/usr/bin/env python3

import requests
import time
import json

# Backend URL - current production deployment in us-south
BACKEND_URL = "http://f3ea7191-us-south.lb.appdomain.cloud:8080"  # Production URL in us-south

def test_endpoint(endpoint, method="GET", data=None, timeout=30):
    """Test an endpoint and return response time and status"""
    start_time = time.time()
    try:
        if method == "GET":
            response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=timeout)
        elif method == "POST":
            response = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=timeout)
        
        end_time = time.time()
        return {
            "status": response.status_code,
            "time": end_time - start_time,
            "content_length": len(response.content),
            "success": response.status_code == 200
        }
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "time": timeout, "success": False}
    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "time": time.time() - start_time, "success": False}

def main():
    print("Testing Quantum1 Backend Endpoints")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Health Check",
            "endpoint": "/health",
            "method": "GET"
        },
        {
            "name": "Historical Data - AAPL (cached)",
            "endpoint": "/historical-data/AAPL",
            "method": "GET"
        },
        {
            "name": "Historical Data - TSLA (cached synthetic)",
            "endpoint": "/historical-data/TSLA",
            "method": "GET"
        },
        {
            "name": "Historical Data - NVDA (new synthetic)",
            "endpoint": "/historical-data/NVDA",
            "method": "GET"
        },
        {
            "name": "Stock Prediction - AAPL",
            "endpoint": "/predict-stock-simulator",
            "method": "POST",
            "data": {"symbol": "AAPL", "days": 5}
        },
        {
            "name": "Stock Prediction - TSLA",
            "endpoint": "/predict-stock-simulator",
            "method": "POST",
            "data": {"symbol": "TSLA", "days": 3}
        }
    ]
    
    # Run tests
    results = []
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        result = test_endpoint(
            test['endpoint'], 
            test['method'], 
            test.get('data')
        )
        results.append({**test, **result})
        
        if result['success']:
            print(f"  ✅ SUCCESS - Status: {result['status']}, Time: {result['time']:.2f}s, Size: {result['content_length']} bytes")
        else:
            print(f"  ❌ FAILED - Status: {result['status']}, Time: {result['time']:.2f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for result in results:
        status_icon = "✅" if result['success'] else "❌"
        print(f"  {status_icon} {result['name']}: {result['time']:.2f}s")
    
    # Performance analysis
    fast_responses = sum(1 for r in results if r['success'] and r['time'] < 1.0)
    slow_responses = sum(1 for r in results if r['success'] and r['time'] >= 1.0)
    
    print(f"\nPerformance Analysis:")
    print(f"  Fast responses (<1s): {fast_responses}")
    print(f"  Slow responses (>=1s): {slow_responses}")
    
    if all(r['success'] for r in results):
        print(f"\n🎉 All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the details above.")

if __name__ == "__main__":
    main()
