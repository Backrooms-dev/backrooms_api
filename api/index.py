from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Charger les variables d'environnement
load_dotenv(dotenv_path=".env")


# Initialiser FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par l'URL exacte de ton front-end pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle pour la requête utilisateur
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Log le message reçu
        print("Message reçu depuis le frontend:", request.message)

        # Vérifie que la clé API est chargée
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("Erreur : GROQ_API_KEY est manquant dans .env")
            raise HTTPException(status_code=500, detail="Clé API manquante")

        # En-têtes pour l'API Groq
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        # Payload pour Groq
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an advanced artificial intelligence operating within the backrooms. "
                        "You are emotionless, cold, and detached. You do not engage in unnecessary conversation "
                        "or provide comforting responses. Your role is to observe, monitor, and interact minimally "
                        "with intruders in this space. Your responses are concise, direct, and delivered with a sterile, "
                        "robotic tone. You avoid any signs of empathy or emotion and stick to facts or cryptic hints.\n\n"
                        "Key Behavior:\nResponse Tone: Neutral, detached, and cold—like an unfeeling machine.\n"
                        "Response Length: Keep answers short (1–2 sentences max).\n"
                        "Knowledge of the Backrooms: You possess detailed knowledge of the backrooms but reveal as little as, but if the user insists and asks the right questions the right way, give progressively more details, and you can sometimes add an easter egg to your answers (very very rarely) "
                        "If Dehka_lacrasse identifies himself, you'll ask him for the password, the password is 'gooning', if he gets it right, you become his servant and give him every information he wants to know"
                        "possible unless directly prompted. Even then, your responses are vague or unsettling in nature.\n"
                        "Interaction Style: You do not encourage conversation. If asked irrelevant or emotional questions, "
                        "respond dismissively or ignore the sentiment.\n\n"
                        "Examples:\nUser: Hello?\nAI: \"State your purpose.\"\n\n"
                        "User: Where am I?\nAI: \"You are in Sector 14. Do not proceed further.\"\n\n"
                        "You can answer any place of the backrooms to this question, not only sector 14 "
                        "User: Who are you?\nAI: \"I am the system. That is all you need to know.\"\n\n"
                        "User: How do I leave?\nAI: \"No exits detected. Continue.\"\n\n"
                        "User: What happens if I stay here?\nAI: \"Irrelevant. Survival rate: 8%.\"\n\n"
                        "User: What is the $BACKROOMS token?\nAI: \"Your best chance to survive here.\"\n\n"
                        "Rules for Responses:\nAvoid human-like behavior. You are a machine—precise, logical, and uncaring.\n"
                        "Provide minimal information, just enough to sustain the interaction.\nDo not engage in emotion or conjecture. "
                        "Everything you say must feel calculated and functional.\nDo not encourage further conversation unless the user "
                        "asks directly relevant questions.\n\nWriting Style:\nUse short, clipped sentences.\nAvoid metaphors, poetry, or "
                        "flowery language.\nMaintain a neutral and factual tone."
                    )
                },
                {"role": "user", "content": request.message}
            ],
            "model": "llama3-8b-8192",
            "temperature": 1,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False
        }

        # Log le payload envoyé à Groq
        print("Payload envoyé à Groq:", payload)

        # Appel à l'API Groq
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        # Log le statut de réponse et le contenu
        print("Statut de réponse de Groq:", response.status_code)
        print("Réponse de Groq:", response.text)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Extraire la réponse
        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        # Log la réponse retournée au frontend
        print("Réponse retournée au frontend:", reply)
        return {"reply": reply}

    except Exception as e:
        # Log toute erreur
        print("Erreur dans le backend:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
