# src/human_review.py
import json
from datetime import datetime

def human_review(decision, explanations):
    print("\n" + "="*60)
    print("👤 HUMAN SUPERVISOR REVIEW")
    print("="*60)
    print(f"AI Recommendation: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print("\n📖 Explanations:")
    print(explanations['summary'])

    while True:
        choice = input("\nDo you Approve, Reject, or Modify this recommendation? (A/R/M): ").strip().upper()
        if choice == 'A':
            status = 'approved'
            modified_action = None
            feedback = input("Optional feedback (press Enter to skip): ").strip()
            break
        elif choice == 'R':
            status = 'rejected'
            modified_action = None
            feedback = input("Reason for rejection (optional): ").strip()
            break
        elif choice == 'M':
            status = 'modified'
            modified_action = input("Enter your modified action: ").strip()
            feedback = input("Reason for modification (optional): ").strip()
            break
        else:
            print("Invalid choice. Please enter A, R, or M.")

    record = {
        'timestamp': datetime.now().isoformat(),
        'ai_recommendation': decision['action'],
        'ai_reasoning': decision['reasoning'],
        'ai_confidence': decision['confidence'],
        'human_status': status,
        'modified_action': modified_action,
        'feedback': feedback
    }
    with open('human_decisions.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
    print("\n✅ Human decision recorded.")
    return record