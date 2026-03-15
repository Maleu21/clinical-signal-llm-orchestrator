#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Clinical Signal LLM Orchestrator ===${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker non installe. Voir https://www.docker.com${NC}"
    exit 1
fi

echo -e "${GREEN}Demarrage des services...${NC}"
docker-compose up --build -d

echo -e "${YELLOW}Attente de l'API...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo -e "${GREEN}API prete${NC}"
        break
    fi
    sleep 2
done

echo -e "${YELLOW}Attente d'Ollama...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo -e "${GREEN}Ollama pret${NC}"
        break
    fi
    sleep 2
done

if ! docker-compose exec ollama ollama list | grep -q mistral; then
    echo -e "${YELLOW}Telechargement de Mistral (4.1 Go, une seule fois)...${NC}"
    docker-compose exec ollama ollama pull mistral
fi
echo -e "${GREEN}Mistral pret${NC}"

if [ ! -f "data/processed/mitbih_windows.npz" ]; then
    echo -e "${YELLOW}Preparation des donnees MIT-BIH (premiere fois)...${NC}"
    docker-compose exec ecg-api python3 -m src.prepare_data
    echo -e "${GREEN}Donnees pretes${NC}"
fi

if [ ! -f "models/ecg_cnn.pt" ]; then
    echo -e "${YELLOW}Entrainement du CNN (premiere fois, ~5 min)...${NC}"
    docker-compose exec ecg-api python3 -m src.train --epochs 5
    echo -e "${GREEN}Modele entraine${NC}"
fi

echo -e "${GREEN}"
echo "================================================"
echo "  Tout est pret !"
echo "  API  -> http://localhost:8000"
echo "  Docs -> http://localhost:8000/docs"
echo "================================================"
echo -e "${NC}"