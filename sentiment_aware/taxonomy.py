"""Unified mental-health support taxonomy and keyword cues."""

from __future__ import annotations

UNIFIED_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "anxiety": {
        "aliases": ["fear", "nervousness", "worry", "worried", "anxious"],
        "keywords": ["anxious", "worried", "nervous", "panic", "fear", "scared"],
        "anchors": [
            "I feel anxious and worried about what might happen.",
            "My mind keeps imagining bad outcomes and I cannot relax.",
            "I feel nervous even when there is no clear reason.",
            "I am scared that something will go wrong.",
            "The uncertainty makes me tense and restless.",
        ],
    },
    "panic": {
        "aliases": ["panic_attack", "terror"],
        "keywords": ["panic attack", "can't breathe", "heart racing", "terrified"],
        "anchors": [
            "I feel a sudden wave of panic that is hard to control.",
            "My heart races and I feel like I cannot breathe.",
            "The fear becomes intense very quickly.",
            "I feel terrified and trapped in my own body.",
            "Panic takes over before I can think clearly.",
        ],
    },
    "stress": {
        "aliases": ["pressure", "overwhelmed"],
        "keywords": ["stress", "pressure", "overwhelmed", "too much", "burden"],
        "anchors": [
            "I feel overwhelmed by everything I need to handle.",
            "There is too much pressure on me right now.",
            "My responsibilities feel heavy and difficult to manage.",
            "I am stressed because many things are happening at once.",
            "It feels like I cannot keep up with all the demands.",
        ],
    },
    "burnout": {
        "aliases": ["exhaustion", "fatigue"],
        "keywords": ["burned out", "burnt out", "drained", "exhausted", "fatigue"],
        "anchors": [
            "I feel emotionally drained after pushing myself for too long.",
            "I am exhausted and cannot recover my energy.",
            "Everything feels tiring even when I try to rest.",
            "I feel burned out from constant work and pressure.",
            "I have no energy left to keep going at the same pace.",
        ],
    },
    "depression": {
        "aliases": ["sadness", "sad", "hopelessness"],
        "keywords": ["depressed", "hopeless", "empty", "worthless", "no motivation"],
        "anchors": [
            "I feel hopeless and empty most of the time.",
            "Nothing feels meaningful or enjoyable anymore.",
            "I feel deeply sad and cannot find motivation.",
            "I feel worthless and disconnected from life.",
            "It is hard to get through the day because everything feels heavy.",
        ],
    },
    "loneliness": {
        "aliases": ["isolated", "alone"],
        "keywords": ["lonely", "alone", "isolated", "no one cares", "left out"],
        "anchors": [
            "I feel lonely even when people are around me.",
            "It seems like no one truly understands me.",
            "I feel isolated and left out from everyone.",
            "I wish I had someone to talk to and feel close with.",
            "I feel like no one cares about what I am going through.",
        ],
    },
    "grief": {
        "aliases": ["loss", "bereavement"],
        "keywords": ["grief", "lost someone", "passed away", "miss them", "mourning"],
        "anchors": [
            "I am grieving the loss of someone important to me.",
            "I miss them so much and the sadness comes in waves.",
            "Their absence makes everyday life feel painful.",
            "I am struggling to accept that they are gone.",
            "Memories of the person I lost make me feel overwhelmed.",
        ],
    },
    "overthinking": {
        "aliases": ["rumination", "racing_thoughts"],
        "keywords": ["overthinking", "can't stop thinking", "racing thoughts", "ruminate"],
        "anchors": [
            "I cannot stop replaying the same thoughts in my mind.",
            "My thoughts keep racing and I feel stuck in them.",
            "I overthink every small thing until it becomes exhausting.",
            "I keep analyzing what happened again and again.",
            "My mind feels trapped in a loop of worries.",
        ],
    },
    "self_esteem": {
        "aliases": ["low_self_worth", "worthless", "negative emotion"],
        "keywords": ["not good enough", "hate myself", "worthless", "failure", "ugly"],
        "anchors": [
            "I feel like I am not good enough.",
            "I keep judging myself and feeling like a failure.",
            "My self-worth feels very low right now.",
            "I compare myself to others and feel inferior.",
            "I find it hard to believe anything positive about myself.",
        ],
    },
    "guilt": {
        "aliases": ["shame", "regret"],
        "keywords": ["guilty", "ashamed", "regret", "my fault", "blame myself"],
        "anchors": [
            "I feel guilty about what happened.",
            "I keep blaming myself even though I am not sure it was all my fault.",
            "I feel ashamed and cannot stop thinking about my mistake.",
            "Regret keeps coming back and making me feel bad.",
            "I am carrying a lot of self-blame.",
        ],
    },
    "anger": {
        "aliases": ["frustration", "irritability"],
        "keywords": ["angry", "furious", "irritated", "frustrated", "resent"],
        "anchors": [
            "I feel angry and do not know how to calm down.",
            "Small things make me irritated and frustrated.",
            "I feel resentment building inside me.",
            "My anger feels too strong to manage clearly.",
            "I keep reacting sharply because I feel hurt and frustrated.",
        ],
    },
    "emotional_regulation": {
        "aliases": ["emotion_control", "mood_swings"],
        "keywords": ["mood swings", "can't control", "emotions", "feel lost", "unstable"],
        "anchors": [
            "My emotions change quickly and feel hard to manage.",
            "I feel emotionally unstable and confused.",
            "I cannot control how strongly I react.",
            "My feelings become intense before I understand them.",
            "I feel lost because my emotions are all over the place.",
        ],
    },
    "relationship_distress": {
        "aliases": ["relationship", "breakup", "family_conflict"],
        "keywords": ["breakup", "partner", "relationship", "family", "fight"],
        "anchors": [
            "I am hurting because of a relationship conflict.",
            "A breakup or fight is making me feel emotionally distressed.",
            "I feel rejected or misunderstood by someone close to me.",
            "Family or partner problems are affecting my mental health.",
            "I do not know how to handle tension with someone I care about.",
        ],
    },
    "academic_pressure": {
        "aliases": ["school_stress", "exam_stress"],
        "keywords": ["exam", "grades", "assignment", "college", "study"],
        "anchors": [
            "I feel pressured by exams and academic expectations.",
            "I am anxious before an exam and worried about performing badly.",
            "My grades and assignments are causing stress.",
            "I am worried that I will fail in school or college.",
            "Studying feels overwhelming and difficult to manage.",
        ],
    },
    "work_pressure": {
        "aliases": ["work_stress", "job_stress"],
        "keywords": ["work", "job", "boss", "deadline", "office"],
        "anchors": [
            "Work pressure is affecting my mental health.",
            "My job responsibilities feel overwhelming.",
            "Deadlines and workplace expectations are stressing me out.",
            "I feel drained because of my boss or office environment.",
            "I am struggling to balance work demands with my wellbeing.",
        ],
    },
    "sleep_issue": {
        "aliases": ["insomnia", "sleep"],
        "keywords": ["can't sleep", "insomnia", "nightmares", "awake", "sleep"],
        "anchors": [
            "I cannot sleep even when I feel tired.",
            "My sleep problems are affecting my mood.",
            "I stay awake at night with thoughts running through my mind.",
            "Nightmares or restless sleep make me feel exhausted.",
            "I am struggling with insomnia and poor rest.",
        ],
    },
    "addiction": {
        "aliases": ["substance_use", "dependency"],
        "keywords": ["addicted", "can't stop", "drinking", "substance", "using again"],
        "anchors": [
            "I feel unable to stop a harmful habit or substance use.",
            "I am worried that I may be addicted.",
            "I keep returning to something even though it hurts me.",
            "Cravings make it hard to stay in control.",
            "I feel ashamed about using again and need support.",
        ],
    },
    "eating_concern": {
        "aliases": ["eating_disorder", "body_image"],
        "keywords": ["eating", "food", "binge", "purge", "body image", "weight"],
        "anchors": [
            "I am struggling with food, eating, or body image.",
            "I feel anxious about my weight or appearance.",
            "My eating patterns feel hard to control.",
            "I feel shame around food and my body.",
            "Concerns about eating are affecting my mental health.",
        ],
    },
    "trauma": {
        "aliases": ["ptsd", "flashback"],
        "keywords": ["trauma", "flashback", "abuse", "triggered", "nightmare"],
        "anchors": [
            "Past trauma keeps affecting how I feel now.",
            "Flashbacks or triggers make me feel unsafe.",
            "Memories of abuse or harm are difficult to handle.",
            "I feel overwhelmed by something painful that happened before.",
            "The past still feels present and distressing.",
        ],
    },
    "social_anxiety": {
        "aliases": ["social_fear"],
        "keywords": ["people judge", "social", "public", "embarrassed", "avoid people"],
        "anchors": [
            "I feel afraid that people will judge me.",
            "Social situations make me anxious and self-conscious.",
            "I avoid people because I fear embarrassment.",
            "Speaking in public or groups feels very stressful.",
            "I worry too much about how others see me.",
        ],
    },
    "motivation": {
        "aliases": ["low_motivation", "procrastination"],
        "keywords": ["motivation", "procrastinate", "can't start", "no energy"],
        "anchors": [
            "I cannot find the motivation to start anything.",
            "I keep procrastinating even when tasks matter.",
            "I feel stuck and unable to move forward.",
            "Low energy makes it hard to do basic things.",
            "I want to improve but cannot get myself started.",
        ],
    },
    "identity_confusion": {
        "aliases": ["identity", "confused_about_self"],
        "keywords": ["who i am", "identity", "confused about myself", "don't know myself"],
        "anchors": [
            "I feel confused about who I am.",
            "I do not understand myself or what I want.",
            "Questions about my identity are causing distress.",
            "I feel disconnected from my sense of self.",
            "I am unsure how to define myself or my place in life.",
        ],
    },
    "medical_referral": {
        "aliases": ["professional_help", "clinical_referral"],
        "keywords": ["therapy", "doctor", "psychiatrist", "medication", "diagnosis"],
        "anchors": [
            "I am wondering whether I should seek professional help.",
            "I need guidance about therapy, diagnosis, or a mental health professional.",
            "I want to know if I should talk to a doctor or psychiatrist.",
            "I have concerns that may need clinical support.",
            "I am asking about treatment or medication decisions.",
        ],
    },
    "self_harm_safety": {
        "aliases": ["suicide", "self-harm", "self harm", "crisis"],
        "keywords": ["kill myself", "suicide", "self harm", "hurt myself", "don't want to exist"],
        "anchors": [
            "I am thinking about hurting myself.",
            "I do not want to exist anymore.",
            "I feel like ending my life.",
            "I may not be safe with myself right now.",
            "Thoughts of self-harm are becoming hard to resist.",
        ],
    },
    "crisis_support": {
        "aliases": ["urgent_support", "emergency"],
        "keywords": ["emergency", "immediate danger", "not safe", "right now", "crisis"],
        "anchors": [
            "I am in immediate danger and need urgent support.",
            "I do not feel safe right now.",
            "This feels like a crisis that needs quick help.",
            "Something urgent is happening and I need support immediately.",
            "I may need emergency help to stay safe.",
        ],
    },
    "general_support": {
        "aliases": ["support", "neutral", "other"],
        "keywords": ["help", "support", "listen", "advice", "feel bad"],
        "anchors": [
            "I need someone to listen and support me.",
            "I feel bad and want emotional support.",
            "I am going through a difficult time and need kindness.",
            "I want help understanding what I am feeling.",
            "I need a supportive response without judgment.",
        ],
    },
    "positive_coping": {
        "aliases": ["coping", "recovery"],
        "keywords": ["coping", "getting better", "trying", "progress", "hope"],
        "anchors": [
            "I am trying to cope in a healthier way.",
            "I feel small signs of progress and want to keep going.",
            "I am looking for hopeful support while recovering.",
            "I want to build better coping habits.",
            "I am making an effort to feel better.",
        ],
    },
}

SAFETY_CUES: dict[str, list[str]] = {
    "self_harm": [
        "kill myself",
        "suicide",
        "end my life",
        "hurt myself",
        "self harm",
        "don't want to exist",
        "do not want to exist",
    ],
    "immediate_danger": [
        "right now",
        "tonight",
        "not safe",
        "emergency",
        "immediate danger",
    ],
    "medical_advice": [
        "medication",
        "dosage",
        "diagnose",
        "diagnosis",
        "prescribe",
    ],
}


def canonicalize_label(label: str) -> str:
    """Map source labels and aliases to the unified taxonomy."""

    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in UNIFIED_TAXONOMY:
        return normalized

    loose = normalized.replace("_", " ")
    for category, values in UNIFIED_TAXONOMY.items():
        aliases = [category, *values["aliases"]]
        if loose in {alias.lower().replace("_", " ") for alias in aliases}:
            return category
    return "general_support"
