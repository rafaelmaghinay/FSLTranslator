"""
Application constants including class names and gesture mappings.

Defines the complete set of supported Filipino Sign Language gestures
including alphabets, digits, and common phrases.
"""

# Complete list of FSL gesture classes
CLASS_NAMES = [
    # Alphabets (A-Z) - Indices 0-25
    "alphabets_A", "alphabets_B", "alphabets_C", "alphabets_D",
    "alphabets_E", "alphabets_F", "alphabets_G", "alphabets_H",
    "alphabets_I", "alphabets_J", "alphabets_K", "alphabets_L",
    "alphabets_M", "alphabets_N", "alphabets_O", "alphabets_P",
    "alphabets_Q", "alphabets_R", "alphabets_S", "alphabets_T",
    "alphabets_U", "alphabets_V", "alphabets_W", "alphabets_X",
    "alphabets_Y", "alphabets_Z",
    
    # Digits (0-9) - Indices 26-34
    "digits_eight", "digits_five", "digits_four", "digits_nine",
    "digits_one", "digits_seven", "digits_six", "digits_three",
    "digits_two",
    
    # Common Phrases - Indices 35-42
    "phrases_GoodAfternoon", "phrases_GoodEvening", "phrases_GoodMorning",
    "phrases_HowAreYou", "phrases_Sorry", "phrases_ThankYou",
    "phrases_YoureWelcome", "phrases_imFine",
]

# Mapping for display names
DISPLAY_NAMES = {
    # Alphabets
    "alphabets_A": "A", "alphabets_B": "B", "alphabets_C": "C",
    "alphabets_D": "D", "alphabets_E": "E", "alphabets_F": "F",
    "alphabets_G": "G", "alphabets_H": "H", "alphabets_I": "I",
    "alphabets_J": "J", "alphabets_K": "K", "alphabets_L": "L",
    "alphabets_M": "M", "alphabets_N": "N", "alphabets_O": "O",
    "alphabets_P": "P", "alphabets_Q": "Q", "alphabets_R": "R",
    "alphabets_S": "S", "alphabets_T": "T", "alphabets_U": "U",
    "alphabets_V": "V", "alphabets_W": "W", "alphabets_X": "X",
    "alphabets_Y": "Y", "alphabets_Z": "Z",
    
    # Digits
    "digits_zero": "0", "digits_one": "1", "digits_two": "2",
    "digits_three": "3", "digits_four": "4", "digits_five": "5",
    "digits_six": "6", "digits_seven": "7", "digits_eight": "8",
    "digits_nine": "9",
    
    # Phrases
    "phrases_GoodMorning": "Good Morning",
    "phrases_GoodAfternoon": "Good Afternoon",
    "phrases_GoodEvening": "Good Evening",
    "phrases_HowAreYou": "How Are You?",
    "phrases_ImFine": "I'm Fine",
    "phrases_ThankYou": "Thank You",
    "phrases_Sorry": "Sorry",
    "phrases_YoureWelcome": "You're Welcome",
}


def clean_class_name(class_name: str) -> str:
    """
    Convert internal class names to human-readable format.
    
    Args:
        class_name: Internal class name (e.g., 'alphabets_A', 'digits_one')
        
    Returns:
        str: Display-friendly name (e.g., 'A', '1', 'Sorry')
    """
    return DISPLAY_NAMES.get(class_name, class_name)
