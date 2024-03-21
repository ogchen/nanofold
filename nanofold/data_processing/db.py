import pymongo
from pymongo import MongoClient


class DBManager:
    def __init__(self, uri):
        self.client = MongoClient(uri)
        try:
            self.client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise RuntimeError(f"Could not connect to MongoDB at {uri}")
        self.db = self.client.data_processing

    def insert_processed_mmcif_files(self, ids):
        self.db.processed_mmcif_files.insert_many(ids)

    def find_processed_mmcif_files(self):
        return self.db.processed_mmcif_files.find()

    def insert_chains(self, chains):
        self.db.chains.insert_many(chains)
