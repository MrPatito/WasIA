# WhatsApp-integrated RAG System

A Retrieval-Augmented Generation system integrated with WhatsApp, built using LangChain, Neo4j, and FastAPI.

## Getting Started

1. Clone the repository
2. Install Docker and Docker Compose
3. Build and run the containers:
   ```bash
   docker-compose up --build
   ```

4. The API will be available at http://localhost:8000
5. Neo4j browser will be available at http://localhost:7474

## Project Structure

```
.
├── src/
│   ├── api/        # FastAPI application
│   ├── core/       # Core RAG functionality
│   ├── database/   # Database models and connections
│   └── processing/ # Document processing pipeline
├── tests/          # Test suite
├── docker/         # Docker configuration files
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

## Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Testing

Run tests with:
```bash
pytest
```
