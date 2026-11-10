from ingestion import rebuild_knowledge_base


class DocumentService:

    def ingest_documents(self):

        total_chunks = rebuild_knowledge_base()

        return {
            "total_chunks": total_chunks
        }