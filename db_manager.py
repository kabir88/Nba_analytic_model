import pymongo
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string="mongodb+srv://ks1751:Olaoluwa88@cluster0.kyiza.mongodb.net/test"):
        try:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client["nba_analytics"]
            logger.info("Connected to MongoDB database")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise 
    def store_data(self, collection_name, data, identifier=None):
        try:
            collection = self.db[collection_name]
            
            # Convert DataFrame to list of dictionaries if needed
            if isinstance(data, pd.DataFrame):
                records = data.to_dict("records")
            elif isinstance(data, dict):
                records = [data]
            else:
                records = data
                
            # Add timestamp
            for record in records:
                record['updated_at'] = datetime.now()
                
            # If identifier is provided, use upsert
            if identifier:
                for record in records:
                    query = {}
                    for key in identifier:
                        if key in record:
                            query[key] = record[key]
                    
                    if query:
                        collection.update_one(query, {"$set": record}, upsert=True)
                    else:
                        collection.insert_one(record)
            else:
                # Bulk insert
                if records:
                    collection.insert_many(records)
                    
            logger.info(f"Stored {len(records)} records in collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error storing data in collection {collection_name}: {e}")
            return False
    
    def retrieve_data(self, collection_name, query=None, projection=None):
        try:
            collection = self.db[collection_name]
            
            if query is None:
                query = {}
                
            cursor = collection.find(query, projection)
            data = list(cursor)
            
            for doc in data:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                    
            logger.info(f"Retrieved {len(data)} records from collection {collection_name}")
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving data from collection {collection_name}: {e}")
            return pd.DataFrame()
    
    def check_data_exists(self, collection_name, query):
        try:
            collection = self.db[collection_name]
            count = collection.count_documents(query)
            return count > 0
        except Exception as e:
            logger.error(f"Error checking data in collection {collection_name}: {e}")
            return False
    
    def delete_data(self, collection_name, query):
        try:
            collection = self.db[collection_name]
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} records from collection {collection_name}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting data from collection {collection_name}: {e}")
            return 0
    
    def close_connection(self):
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")


            