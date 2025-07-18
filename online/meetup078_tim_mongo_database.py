#!/usr/bin/env python3
"""MeetUp 078 - Beginners' Python and Machine Learning - 22 Sep 2020 - Mongo database

Youtube: https://youtu.be/ZkulTVfpkuo
Colab:   https://colab.research.google.com/drive/1Ct8EZCH9Prt_bFwQ5hniXUglz545zG3A
Github:  https://github.com/timcu/bpaml-sessions/tree/master/online
MeetUp:  https://www.meetup.com/beginners-python-machine-learning/events/273223520/

Databases allow you to save data when your Python program is not running. They are much faster, more powerful and more memory
efficient than data files for retrieving and updating specific data within the data set.

Mongo DB is a schema-less/NoSQL/document database which stores data as collections of documents compared to SQL databases which
store tables of data. Think of a document as a Python `dict`. NoSQL is now said to mean "not only SQL" because they can support
SQL-like queries.

Learning objectives:
- Create, Read, Update, Delete data in a Mongo database (CRUD)
- Filters
- Indexes
- Aggregates

For this workshop you will need a MongoDB database you can connect to from your Python script. I suggest a free database at
https://www.mongodb.com/cloud/atlas/

- Create a free shared cluster (group of servers hosting database mirrors)
 - AWS Sydney ap-southeast-2
 - M0 Sandbox - Shared RAM, 512MB storage - Free forever
 - Cluster name - Cluster0
- Create first database user (Database Access):
 - privileges - read and write to any database
- Whitelist IP address (Network access):
 - allow from anywhere
- Get uri for connection by application:
 - `mongodb+srv://<username>:<password>@cluster0.dfwz3.mongodb.net/<dbname>?retryWrites=true&w=majority`

### References

Mongo DB : https://docs.mongodb.com/manual/reference/

PyMongo: https://pymongo.readthedocs.io/en/stable/

@author D Tim Cummings
"""

# URI starts with mongodb+srv which means it uses DNS SRV and TXT entries for
# cluster0.dfwz3.mongodb.net to determine the full network configuration of the
# cluster so we don't have to specify full cluster in URI
# However, we do need to install dnspython alongside pymongo.
# Uncomment next line to install in Google colab
# !pip install pymongo[srv]
# Alternatively add the following two lines to requirements.txt without # signs
# pymongo
# dnspython

# Standard libraries we will be using
import json
from pprint import pprint
from urllib.request import urlopen
import re
import html
import time
# Third party library which we installed above
import pymongo

DBNAME = "db_bpaml078"
try:
    with open("mongopw.txt", "r") as pw:
        # password is stored in a file so not visible in code
        # remember to upload file before running script
        PASSWORD = pw.read()
except FileNotFoundError:
    print("You need to create a file mongopw.txt containing your mongo password before running this script")
    exit(1)
# Create the URI using format str
USERNAME = "my_username"  # CHANGE THIS LINE TO YOUR DATABASE ACCESS USERNAME
# CHANGE NEXT LINE TO URI FOR YOUR DATABASE. HOSTNAME cluster0.dfwz3.mongodb.net SHOULD CHANGE
MDB_URI = f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.dfwz3.mongodb.net/{DBNAME}?retryWrites=true&w=majority"

client = pymongo.MongoClient(MDB_URI)  # new instance of class so uses parentheses ()
db = client[DBNAME]  # index DBNAME so use square brackets []. Could also use attribute notation client.db_bpaml078
try:
    list_database_names = client.list_database_names()
except pymongo.errors.OperationFailure:
    print("Edit MDB_URI and USERNAME with your own URI before running this script")
    exit(1)
print("Database names in Mongo. Databases are not created until required.")
print(client.list_database_names())

# How mongodb+srv uses DNS to determine how to connect to database
# use dig on mac or linux
# dig _mongodb._tcp.cluster0.dfwz3.mongodb.net SRV
# dig cluster0.dfwz3.mongodb.net TXT
# online dns lookup https://www.ultratools.com/tools/dnsLookup

# Clear out collections from previous sessions. Requires write access.
db.drop_collection("clct_first")
db.drop_collection("clct_federal_mp")
db.drop_collection("clct_event")

# Use list_collection_names() to find the current collections (tables)
if len(db.list_collection_names()) == 0:
    print("There are no collections yet")
else:
    print("Collection names")
    print(db.list_collection_names())

# How to create new data in a collection. Collection is created automatically
# Can use attribute notation db.clct_first or index notation db['clct_first']
result = db.clct_federal_mp.insert_one(
    {"name": "Scott Morrison", "party": "Liberal Party of Australia", "position": "Prime Minister"})
print("Inserted one document and the id was", result.inserted_id)

print("Collection names after inserting one document. Notice how collection automatically created")
print(db.list_collection_names())

print("Documents in the collection")
docs = db.clct_federal_mp.find()
for doc in docs:
    print(doc)

# what datatypes can be inserted
try:
    result = db.clct_federal_mp.insert_one("Labor")
    print(f"Successfully inserted a str. Result {result.inserted_id}")
except TypeError as te:
    print("There was a TypeError inserting a str")
    print(te)

# how to insert several records at once
lst_member = [
    {"name": "Josh Frydenberg", "party": "Liberal Party of Australia", "position": "Treasurer"},
    {"name": "Anthony Albanese", "party": "Australian Labor Party", "position": "Leader of the Opposition"},
    {"name": "Julian Simmonds", "party": "Liberal National Party of Queensland"},
    {"name": "Michael McCormack", "party": "The Nationals",
     "position": ["Deputy Prime Minister", "Minister for Infrastructure, Transport and Regional Development"]},
]
result = db.clct_federal_mp.insert_many(lst_member)
print("Inserted IDs when inserting many documents")
for r in result.inserted_ids:
    print(r)

print("Documents in the collection. Notice that unspecified _ids are created as ObjectIds")
docs = db.clct_federal_mp.find()
for doc in docs:
    print(doc)

# Load the json data from the Brisbane City Council url into a Python list of dicts
url = "http://www.trumba.com/calendars/brisbane-city-council.json"
events = json.load(urlopen(url=url))
print("First event from Brisbane City Council JSON events data")
pprint(events[0])

# Create new collection and insert data
result = db.clct_event.insert_many(events)
print(f"Number of events inserted into mongo database {len(result.inserted_ids)}")
pprint(result.inserted_ids[:10])

# Find first all day event
# SQLite equivalent
# select * from tbl_event where allDay is TRUE limit 1;
event = db.clct_event.find_one({"allDay": True})
# delete some fields so print not too long
del(event['customFields'])
del(event['description'])
print("First allDay event in database (find_one using a filter)")
pprint(event)

# Find all events at location "Museum of Brisbane, Brisbane City"
# SQLite equivalent
# select * from tbl_event where location = "Museum of Brisbane, Brisbane City"
my_filter = {"location": "Museum of Brisbane, Brisbane City"}
result = db.clct_event.find(my_filter)
print(f"result is of type {type(result)}. Cursors step through results without loading all results into memory at once\n")
print(f"Events at {my_filter['location']}\n")
for r in result:
    print(r['location'], r['title'], r['description'])

# Once cursor has traversed results we can't do it again without rerunning query
for r in result:
    print(r['location'], r['description'])
else:
    print("No more once cursor has traversed results")

# Find all events at location containing the word "City" and sort by location in reverse alphabetical order
# SQLite equivalent
# select * from tbl_event where location = "%City%" order by location desc;
my_filter = {"location": re.compile("City")}
# Alternative syntax. Mongodb loves nested dictionaries
# filter = {"location": {"$regex": "City"}}
cursor = db.clct_event.find(filter=my_filter, sort=[("location", -1)])
print("Events in City\n")
for r in cursor:
    print(f"{r['location']:40} {r['title']:40} {r['description']}")

# Task 1: Find all the members of the Australian Labor Party
party = "Australian Labor Party"

# Solution 1
print(f"\nSolution 1: Members of '{party}'")
for r in db.clct_federal_mp.find({"party": party}):
    print(r)

# Task 2: Find all the members of the government
parties = ("Liberal Party of Australia", "The Nationals", "Liberal National Party of Queensland")  # can be list or tuple

# Solution 2
print("\nSolution 2 using $in. Members of government")
# https://docs.mongodb.com/manual/reference/operator/query/in/
# SQLite equivalent
# select * from tbl_first where party in ("Liberal Party of Australia", "The Nationals", "Liberal National Party of Queensland");
my_filter = {"party": {"$in": parties}}
for r in db.clct_federal_mp.find(filter=my_filter):
    print(r)

print("\nSolution 2 using regular expression and limiting returned fields. Members of government")
# projection can be used to restrict the fields returned
# SQLite equivalent
# select name, party from tbl_first where party like "%Liberal%" or party like "%National%";
my_filter = {"party": re.compile("Liberal|National")}
projection = {"_id": 0, "name": 1, "party": 1}
for r in db.clct_federal_mp.find(filter=my_filter, projection=projection):
    print(r)

# Task 3: Extract just the customFields from clct_event

# Solution 3:
print("\nSolution 3: First 10 customFields lists")
cursor = db.clct_event.find(projection={"_id": 0, "customFields": 1})
for r in cursor[:10]:
    print(r['customFields'])
cursor = db.clct_event.find(projection={"_id": 0, "customFields": 1})
lst_custom = []
for r in cursor:
    lst_custom += r['customFields']
print("\nAll custom fields in one list")
for cf in lst_custom[:10]:
    print(cf)

# Above code can be streamlined into one command by aggregating

# examples of aggregate functions
# {"$unwind": "$customFields"},  # takes a list (or array) and expands it one item per record
# {"$match": {"customFields.fieldID": 21859}}  # if you wanted to filter during aggregation process
# {"$project": {"_id": 1, "customFields": 1}},  # which fields you want to keep or remove from results
# {"$group": {"_id": "$_id", "customFields": {"$push": "$customFields"}}}  # opposite of unwind

cursor = db.clct_event.aggregate([
        {"$unwind": "$customFields"},
        {"$project": {"_id": 0, "customFields": 1}},
    ]
)
print("Solution 3 - Using aggregate functions")
# Easier to use a for loop with cursor but if you wanted to use a while loop catch the StopIteration exception
i = 0
try:
    while i < 10:
        i += 1
        r = cursor.next()
        print(i, r)
except StopIteration:
    print("no more results in cursor")

# How to update record
my_filter = {"name": "Julian Simmonds"}
db.clct_federal_mp.update_one(filter=my_filter, update={"$set": {"division": "Ryan"}})
# check the results
for r in db.clct_federal_mp.find():
    print(r)

# Task 4: Create a new description field with unicode instead of html entities (html.unescape() on 'description' field)

print("\nSolution 4: converting html entities to unicode and updating database one record at a time")
# How to update every record in database, one at a time
# Normally you don't make data in one field dependent on data in another field
# When you do it is rare to run such a process on the whole database
print("description fields with html escaped entities\n")
for c in db.clct_event.find({}, limit=5):
    print(c["description"])
start = time.perf_counter()
print(f"\ncreating new fields unescaping the entities start={start:0.2f}")

for r in db.clct_event.find():
    db.clct_event.update_one({"_id": r['_id']}, {"$set": {"description_raw": html.unescape(r["description"])}})

finish = time.perf_counter()
print(f"finished updating database. finish={finish:0.2f} duration={finish-start:0.2f} seconds\n")
print("description fields with html escaped entities converted back to unicode\n")
for c in db.clct_event.find({}, limit=5):
    print(c["description_raw"])

print("\nSolution 4: Update every record in database in bulk")
start = time.perf_counter()
print(f"\ncreating new fields unescaping the entities start={start:0.2f}")

lst_bulk = []
for r in db.clct_event.find():
    lst_bulk.append(pymongo.UpdateOne({"_id": r['_id']}, {"$set": {"description_raw": html.unescape(r["description"])}}))
db.clct_event.bulk_write(lst_bulk)

finish = time.perf_counter()
print(f"finished updating database. finish={finish:0.2f} duration={finish-start:0.2f} seconds\n")
print("description fields with html escaped entities converted back to unicode\n")
for c in db.clct_event.find({}, limit=5):
    print(c["description_raw"])

# Create an index to speed queries and/or enforce uniqueness
db.clct_federal_mp.create_index("name", unique=True)

try:
    db.clct_federal_mp.insert_one({"name": "Scott Morrison"})
except pymongo.errors.DuplicateKeyError as dke:
    print("Duplicate Key Error")
    print(dke)

# Create a text index to improve text searches. Only one text index per collection
db.clct_event.create_index([("title", "text"), ("description_raw", "text")], default_language='english')

r = db.clct_event.find_one({"description_raw": {"$regex": "myths tales"}})
pprint(r)

r = db.clct_event.find_one({"$text": {"$search": "myths tales"}})
pprint(r)

print("Example of using aggregate functions to find count of all locations")
result = db.clct_event.aggregate([
    {"$group": {"_id": "$location", "count": {"$sum": 1}}}
])
i = 0
for r in result:
    i += 1
    print(i, r)
    if i >= 20:
        break
