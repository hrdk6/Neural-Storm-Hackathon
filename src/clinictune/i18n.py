"""Plain-language notes for the screening result.

Hindi and Kannada strings are drafts written for this project; they MUST be reviewed by a native-speaking
clinician before use with patients. ``reviewed`` records that status and is shown in the API response.
"""

from __future__ import annotations

from typing import Any

LANGS: dict[str, dict[str, Any]] = {
    "en": {
        "reviewed": True,
        "condition": {"tb": "possible tuberculosis", "dr": "referable diabetic retinopathy"},
        "refer": "The screening aid found signs that may indicate {condition}. Please refer the patient for "
        "confirmatory testing.",
        "low_risk": "The screening aid did not find signs of {condition} in this image. Continue routine "
        "clinical care; symptoms still need to be assessed.",
        "uncertain": "The screening aid could not reach a confident result. Please ask a senior clinician to "
        "review the image, or retake it.",
        "retake": "The image quality is not good enough to screen. Please retake the image.",
        "not_diagnosis": "This is a screening aid, not a diagnosis.",
    },
    "hi": {
        "reviewed": False,
        "condition": {"tb": "संभावित टीबी (क्षय रोग)", "dr": "रेफ़र करने योग्य डायबिटिक रेटिनोपैथी"},
        "refer": "स्क्रीनिंग सहायक को इस छवि में {condition} के संकेत मिले हैं। कृपया मरीज़ को पुष्टि-जाँच के लिए भेजें।",
        "low_risk": "स्क्रीनिंग सहायक को इस छवि में {condition} के संकेत नहीं मिले। सामान्य क्लिनिकल देखभाल जारी "
        "रखें; लक्षणों की जाँच फिर भी ज़रूरी है।",
        "uncertain": "स्क्रीनिंग सहायक निश्चित परिणाम नहीं दे सका। कृपया वरिष्ठ चिकित्सक से छवि की समीक्षा कराएँ या छवि दोबारा लें।",
        "retake": "छवि की गुणवत्ता स्क्रीनिंग के लिए पर्याप्त नहीं है। कृपया छवि दोबारा लें।",
        "not_diagnosis": "यह एक स्क्रीनिंग सहायक है, निदान नहीं।",
    },
    "kn": {
        "reviewed": False,
        "condition": {"tb": "ಕ್ಷಯರೋಗದ (ಟಿಬಿ) ಸಾಧ್ಯತೆ", "dr": "ಉಲ್ಲೇಖಿಸಬೇಕಾದ ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ"},
        "refer": "ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯಕವು ಈ ಚಿತ್ರದಲ್ಲಿ {condition} ಸೂಚಿಸುವ ಲಕ್ಷಣಗಳನ್ನು ಕಂಡಿದೆ. ದೃಢೀಕರಣ ಪರೀಕ್ಷೆಗಾಗಿ ರೋಗಿಯನ್ನು ಕಳುಹಿಸಿ.",
        "low_risk": "ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯಕವು ಈ ಚಿತ್ರದಲ್ಲಿ {condition} ಲಕ್ಷಣಗಳನ್ನು ಕಾಣಲಿಲ್ಲ. ಸಾಮಾನ್ಯ ಚಿಕಿತ್ಸಾ ಆರೈಕೆಯನ್ನು "
        "ಮುಂದುವರಿಸಿ; ರೋಗಲಕ್ಷಣಗಳನ್ನು ಇನ್ನೂ ಪರಿಶೀಲಿಸಬೇಕು.",
        "uncertain": "ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯಕವು ಖಚಿತ ಫಲಿತಾಂಶ ನೀಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಹಿರಿಯ ವೈದ್ಯರಿಂದ ಚಿತ್ರವನ್ನು ಪರಿಶೀಲಿಸಿ ಅಥವಾ ಚಿತ್ರವನ್ನು ಮತ್ತೆ ತೆಗೆಯಿರಿ.",
        "retake": "ಚಿತ್ರದ ಗುಣಮಟ್ಟ ಸ್ಕ್ರೀನಿಂಗ್‌ಗೆ ಸಾಕಾಗುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ಚಿತ್ರವನ್ನು ಮತ್ತೆ ತೆಗೆಯಿರಿ.",
        "not_diagnosis": "ಇದು ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯಕ, ರೋಗನಿರ್ಣಯವಲ್ಲ.",
    },
}


def note(lang: str, key: str, task: str) -> dict[str, Any]:
    L = LANGS.get(lang, LANGS["en"])
    text = L[key].format(condition=L["condition"][task]) + " " + L["not_diagnosis"]
    return {"lang": lang if lang in LANGS else "en", "text": text, "translation_reviewed": L["reviewed"]}
