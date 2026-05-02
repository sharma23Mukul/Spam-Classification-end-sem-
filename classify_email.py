import sys, os, pickle
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.pipeline import preprocess

with open('models/app_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

m = bundle['multinomial']

msg = sys.argv[1] if len(sys.argv) > 1 else """Hey community,
It's time to check out the latest and greatest in Streamlit. First up, release 1.56 is here with some exciting new additions — a brand-new dropdown widget, native iframe support, and a long-requested filtering upgrade for selects! There's also a new pivot table component to try out, two reads from the team on building apps in the age of AI agents, and a couple big events coming up. Keep reading to learn more!
Release 1.56
Highlights
Introducing st.menu_button, a new dropdown button widget with a customizable popover
Introducing st.iframe: embed external URLs or raw HTML content directly in your app, no custom components API needed
st.selectbox and st.multiselect now support a filter_mode parameter that lets users search and filter options by typing
See the full changelog for much more!
New pivot table component
Working with multi-dimensional data? The new streamlit-pivot component brings spreadsheet-style power straight into your apps! It supports interactive sorting and filtering, subtotals with collapse/expand, conditional formatting, data export, drill-down detail panels, drag-and-drop field configuration, synthetic (derived) measures, date/time hierarchies with period-over-period comparisons, hierarchical row layouts, column resize, fullscreen mode, and server-side pre-aggregation for large datasets.
Learn more in the PyPI and the demo app.
From the blog: building in the age of agents
The way devs build apps is changing fast. See how the team is thinking about this in two new blog posts.
How Agentic Engineering Changed the Way I Build Streamlit Apps: Chanin walks through how he built a full 17-page multi-page data exploration app in about a week using agentic engineering structured planning docs, custom agent skills, and iterative AI-assisted sessions.
The Repo Is the Harness: How We Made an 8-Year-Old Codebase Agent-Native: Lukas shares the Streamlit team's playbook for making the codebase work great with AI coding agents. Features that once took a sprint now ship in hours, and newly reported bugs are at a three-year low.
Upcoming events
Come connect with the team and community at these upcoming events
April 28: Streamlit Virtual Meetup, 9 10am PT
Join Streamlitters from around the world at this global virtual meetup! Hear directly from Lead Developer Advocate Chanin Nantasenamat and Senior Software Engineer Lukas Masuch on building apps using agentic engineering, plus a live Q&A. Tune in on YouTube or LinkedIn. RSVP here to add it to your calendar!
May 13-19: PyCon US, Long Beach, CA
Will you be at PyCon US? If so, be sure to say hello to the team! We'll be at a booth as well hosting a super fun hack night.
June 1-4: Snowflake Summit 2026, San Francisco
Meet fellow data and AI leaders at Snowflake Summit 26! Learn from experts and peers + gain practical skills to build enterprise-ready agentic solutions. Streamlitters get a special $400 discount. Just use the code SUMMIT26-STREAMLIT when you register here.
Join the conversation
Looking for help or want to share a project? Stop by the forum to ask questions, showcase your work, and discover all the cool stuff others are building.
Have a great one!
Best,
Streamlit team"""

tokens = preprocess(msg)
result = m.predict_with_confidence(tokens)

print("=" * 55)
print("  SPAM CLASSIFICATION REPORT")
print("=" * 55)
print(f"  Prediction:  {result['prediction'].upper()}")
print(f"  Confidence:  {result['confidence']*100:.1f}%")
print(f"  P(Spam):     {result['probabilities']['spam']*100:.2f}%")
print(f"  P(Ham):      {result['probabilities']['ham']*100:.2f}%")
print(f"  Tokens:      {len(tokens)} words analyzed")
print(f"  Verdict:     {result['explanation']}")
print(f"  Alpha (α):   {m.alpha}")
print("=" * 55)

# Show top spam-indicator tokens
spam_ll = m.log_likelihoods.get('spam', {})
ham_ll = m.log_likelihoods.get('ham', {})
token_set = set(tokens)
scores = []
for t in token_set:
    if t in spam_ll and t in ham_ll:
        scores.append((t, spam_ll[t] - ham_ll[t]))
scores.sort(key=lambda x: -x[1])

print("\n  Top 10 SPAM-indicating words in this email:")
for w, s in scores[:10]:
    print(f"    {w:<20s}  score: {s:+.4f}")

print("\n  Top 10 HAM-indicating words in this email:")
for w, s in scores[-10:]:
    print(f"    {w:<20s}  score: {s:+.4f}")
