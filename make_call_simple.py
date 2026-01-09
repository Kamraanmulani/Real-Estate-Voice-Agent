"""
Simple Outbound Call Script (WITHOUT Vocode)
Works with twilio_simple_agent.py

Usage:
    python make_call_simple.py
"""
import os
from dotenv import load_dotenv

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("❌ Twilio not installed. Run: pip install twilio")
    exit(1)

load_dotenv()


def make_call(to_number: str):
    """Make an outbound call to Indian mobile"""
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    ngrok_url = os.getenv("NGROK_URL") or os.getenv("BASE_URL")
    
    if not all([account_sid, auth_token, from_number, ngrok_url]):
        print("❌ Missing configuration!")
        print("Required in .env:")
        print("  - TWILIO_ACCOUNT_SID")
        print("  - TWILIO_AUTH_TOKEN")
        print("  - TWILIO_PHONE_NUMBER")
        print("  - NGROK_URL or BASE_URL")
        return
    
    # Ensure ngrok URL doesn't have trailing slash
    ngrok_url = ngrok_url.rstrip('/')
    
    print("\n" + "="*70)
    print("📞 Making Outbound Call (Simple Agent)")
    print("="*70)
    print(f"From: {from_number}")
    print(f"To: {to_number}")
    print(f"Webhook: {ngrok_url}/voice")
    print("="*70 + "\n")
    
    try:
        client = Client(account_sid, auth_token)
        
        call = client.calls.create(
            to=to_number,
            from_=from_number,
            url=f"{ngrok_url}/voice",
            method="POST",
            status_callback=f"{ngrok_url}/status",
            status_callback_method="POST",
            status_callback_event=["initiated", "ringing", "answered", "completed"]
        )
        
        print("✅ Call initiated successfully!")
        print(f"\nCall SID: {call.sid}")
        print(f"Status: {call.status}")
        print(f"\n💡 The phone will ring on {to_number}")
        print("When you answer, Miss Riverwood will greet you!")
        print("Speak in Hindi, Hinglish, or English!")
        print("\n" + "="*70 + "\n")
        
        return call.sid
        
    except Exception as e:
        print(f"\n❌ Failed to make call: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure twilio_simple_agent.py is running")
        print("2. Make sure ngrok is running (ngrok http 5000)")
        print("3. Check Twilio account has sufficient balance")
        print("4. Verify phone number format (+919136812065)")
        print("5. Enable 'Global Calling' in Twilio Console")
        return None


if __name__ == "__main__":
    # YOUR INDIAN MOBILE NUMBER
    TO_NUMBER = "+919136812065"  # Replace with your number
    
    print("\n🚀 Miss Riverwood Simple Call System")
    print("="*70)
    
    if not TO_NUMBER.startswith("+91"):
        print("⚠️  WARNING: Phone number should start with +91 for India")
        print("Example: +919136812065")
        print("\nUpdate TO_NUMBER in this script!")
        exit(1)
    
    call_sid = make_call(TO_NUMBER)
    
    if call_sid:
        print("\n💡 Pro Tip:")
        print("Track your call in Twilio Console:")
        print(f"https://console.twilio.com/us1/monitor/logs/calls/{call_sid}")
