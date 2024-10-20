import os, asyncio
from ingestion.IngestionSvc import IngestionSvc
from ingestion.utils.loader import get_pdf_paths
from ingestion.config.load_config import load_config

def main():
    config = load_config()

    svc = IngestionSvc(embedding_config=config['embeddings'],
                       vectorstore_config=config['vectorstore'],
                       db_dir=config['db_dir'])

    paths = get_pdf_paths(config['pdf_folder'])

    asyncio.run(svc.build_indices_from_documents(paths))

if __name__ == "__main__":
    main()