#!/bin/bash
set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Clinical Signal LLM Orchestrator ===${NC}"

# Charger les variables d'environnement
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODEL=mistral

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker n'est pas installé. Installe Docker Desktop : https://www.docker.com${NC}"
    exit 1
fi

# Vérifier qu'Ollama est installé
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama non trouvé. Installation...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Lancer Ollama selon l'OS
echo -e "${GREEN}▶ Démarrage d'Ollama...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac
    open -a Ollama 2>/dev/null || OLLAMA_HOST=0.0.0.0:11434 ollama serve &
else
    # Linux / WSL
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &
fi

# Attendre qu'Ollama soit prêt
echo -e "${YELLOW}⏳ Attente d'Ollama...${NC}"
for i in {1..15}; do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama prêt${NC}"
        break
    fi
    sleep 1
done

# Vérifier que Mistral est téléchargé
if ! ollama list | grep -q mistral; then
    echo -e "${YELLOW}⏳ Téléchargement de Mistral (4.1 Go, une seule fois)...${NC}"
    ollama pull mistral
fi

echo -e "${GREEN}✅ Mistral prêt${NC}"

# Lancer Docker Compose
echo -e "${GREEN}▶ Démarrage de l'API...${NC}"
docker-compose up --build