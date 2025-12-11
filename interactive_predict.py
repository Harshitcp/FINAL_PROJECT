
import sys
import pandas as pd
from pathlib import Path

# Add the current directory to sys.path to import from predict_and_compare
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from predict_and_compare import predict_single_user
except ImportError as e:
    print(f"Error: Could not import 'predict_single_user' from 'predict_and_compare.py'.")
    print(f"Exception: {e}")
    print("Make sure both files are in the same directory.")
    sys.exit(1)

def get_user_input(prompt, type_func=str, valid_options=None):
    while True:
        try:
            user_input = input(prompt + ": ")
            value = type_func(user_input)
            
            if valid_options and value not in valid_options:
                print(f"Invalid input. Options: {valid_options}")
                continue
                
            return value
        except ValueError:
            print(f"Invalid input type. Expected {type_func.__name__}.")

def get_yes_no(prompt):
    while True:
        user_input = input(prompt + " (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            return 1
        elif user_input in ['n', 'no']:
            return 0
        else:
            print("Please answer 'y' or 'n'.")

def main():
    print("=" * 60)
    print("TWITTER BOT DETECTION - INTERACTIVE PREDICTION")
    print("=" * 60)
    print("Please enter the user's details below:")
    print("-" * 40)

    try:
        user_data = {}
        
        # Numeric features
        user_data['statuses_count'] = get_user_input("Statuses Count (Tweets)", int)
        user_data['followers_count'] = get_user_input("Followers Count", int)
        user_data['friends_count'] = get_user_input("Friends Count (Following)", int)
        user_data['favourites_count'] = get_user_input("Favourites Count (Likes)", int)
        user_data['listed_count'] = get_user_input("Listed Count", int)
        
        # Boolean features
        user_data['default_profile'] = get_yes_no("Has Default Profile?")
        user_data['geo_enabled'] = get_yes_no("Geo Enabled?")
        user_data['profile_use_background_image'] = get_yes_no("Uses Profile Background Image?")
        user_data['verified'] = get_yes_no("Is Verified?")
        
        # Date feature
        while True:
            date_str = input("Account Creation Date (YYYY-MM-DD): ")
            try:
                user_data['created_at'] = pd.Timestamp(date_str)
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
        
        print("\n" + "-" * 60)
        print("Analyzing...")
        
        # Make prediction
        predictions = predict_single_user(user_data)
        
        print("\nPREDICTION RESULTS:")
        print("-" * 60)
        
        for model_name, result in predictions.items():
            prob = result['bot_probability']
            pred = result['prediction']
            
            # Color coding (conceptual)
            symbol = "[BOT] " if result['is_bot'] else "[GENUINE]"
            
            print(f"{symbol:10s} {model_name:20s}: {pred:10s} (Bot Probability: {prob:.2%})")
            
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
