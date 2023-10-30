
import h5py
import numpy as np

class HDF5EmbeddingStorage:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def store_system_role_embedding(self, conversation_id, text, embedding):
        with h5py.File(self.file_path, 'a') as f:
            grp = f.create_group(conversation_id)
            grp.create_dataset("system_role_text", data=text, dtype=h5py.string_dtype(encoding='utf-8'))
            grp.create_dataset("system_role_embedding", data=embedding)
    
    def store_entry(self, conversation_id, entry_id, query_text, query_embedding, response_text, response_embedding, cumulative_text, cumulative_embedding):
        with h5py.File(self.file_path, 'a') as f:
            entries_grp = f.require_group(f"{conversation_id}/entries")
            entry_grp = entries_grp.create_group(entry_id)
            entry_grp.create_dataset("query_text", data=query_text, dtype=h5py.string_dtype(encoding='utf-8'))
            entry_grp.create_dataset("query_embedding", data=query_embedding)
            entry_grp.create_dataset("response_text", data=response_text, dtype=h5py.string_dtype(encoding='utf-8'))
            entry_grp.create_dataset("response_embedding", data=response_embedding)
            entry_grp.create_dataset("cumulative_text", data=cumulative_text, dtype=h5py.string_dtype(encoding='utf-8'))
            entry_grp.create_dataset("cumulative_embedding", data=cumulative_embedding)
    
    def get_system_role(self, conversation_id):
        with h5py.File(self.file_path, 'r') as f:
            grp = f[conversation_id]
            text = grp["system_role_text"][()]
            embedding = grp["system_role_embedding"][:]
            return text, embedding
    
    def get_entry(self, conversation_id, entry_id):
        with h5py.File(self.file_path, 'r') as f:
            entry_grp = f[f"{conversation_id}/entries/{entry_id}"]
            query_text = entry_grp["query_text"][()]
            query_embedding = entry_grp["query_embedding"][:]
            response_text = entry_grp["response_text"][()]
            response_embedding = entry_grp["response_embedding"][:]
            cumulative_text = entry_grp["cumulative_text"][()]
            cumulative_embedding = entry_grp["cumulative_embedding"][:]
            return query_text, query_embedding, response_text, response_embedding, cumulative_text, cumulative_embedding
