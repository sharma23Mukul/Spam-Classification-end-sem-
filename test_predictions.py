import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.pipeline import preprocess

with open('models/app_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

m = bundle['multinomial']

tests = [
    ("Streamlit newsletter (should be spam/promo)", """Hey community, It's time to check out the latest and greatest in Streamlit. First up, release 1.56 is here with some exciting new additions. New pivot table component. Working with multi-dimensional data? The new streamlit-pivot component brings spreadsheet-style power. Join the conversation. Looking for help or want to share a project? Stop by the forum."""),
    ("Buy one get one free (should be spam)", "Anniversary Special: Buy one get one free. As our loyal customer, get exclusive $60 off $75+: example.com/6058 Offer code: WELCOME20."),
    ("Win free cash (should be spam)", "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"),
    ("OTP code (should be ham)", "Your verification code is 483921. Please enter this code to complete your login. This code expires in 10 minutes."),
    ("Forum reply (should be ham)", "Thanks for sharing your experience! I had a similar issue with my setup and found that updating the driver fixed it. Let me know if you need any help."),
    ("Social media notif (should be ham)", "John liked your photo. Sarah commented on your post: 'Great picture!' You have 3 new followers this week."),
    ("Software update (should be ham)", "A new version of the app is available. Version 2.5.1 includes bug fixes and performance improvements. Update now from the settings page."),
]

for name, msg in tests:
    tokens = preprocess(msg)
    result = m.predict_with_confidence(tokens)
    pred = result['prediction'].upper()
    conf = result['confidence'] * 100
    print(f"  [{pred:4s} {conf:5.1f}%] {name}")
