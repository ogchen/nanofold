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

    def processed_mmcif_files(self):
        return self.db.processed_mmcif_files

    def chains(self):
        return self.db.chains
