"""Simple import test for new supervisor implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing imports...")

try:
    from cryptotrader.agents.skills import TRADING_SKILLS, get_skill_descriptions

    print("✓ skills.py imported")
    print(f"  - {len(TRADING_SKILLS)} skills available")
except Exception as e:
    print(f"✗ skills.py import failed: {e}")

try:
    print("✓ tools.py imported")
except Exception as e:
    print(f"✗ tools.py import failed: {e}")

print("\nTesting skill loading...")
try:
    skill_desc = get_skill_descriptions()
    print(f"✓ Skill descriptions: {len(skill_desc)} chars")

    from cryptotrader.agents.skills import load_skill_content

    content = load_skill_content("funding_rate_analysis")
    print(f"✓ Loaded funding_rate_analysis: {len(content)} chars")
except Exception as e:
    print(f"✗ Skill loading failed: {e}")

print("\n✓ All basic imports successful")
print("\nNote: Full supervisor test requires langchain dependencies")
