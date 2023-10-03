My Learnings on [ByteByteGo](https://bytebytego.com/).

# Scale From 0 to Millions of People

1) HTTP is the communication protocol used between web servers and mobile app.

2) Separating database and app tiers help us scale them independently.

3) CouchDB, Neo4j, Hbase are some NoSQL technologies. There are also Cassandra, Couchbase, MongoDB.

4) 4 categories of NoSQL databases:

    - Key-value stores:
    - Graph stores:
    - Column stores:
    - Document stores:

5) For most developers, RDBMS work very well.

6) NoSQL databases might be the right choice if 

    - the app requiring super-low latency.

    - data are unstructured or no relation exists in data.

    - You only need to serialize and deserialize data (JSON, XML, YAML, etc.).

    - Data to store is huge.

7) When traffic is low, vertical scaling is the way to prefer. The real advantage of vertical scaling(scale up) is its simplicity.

8) When Load Balancer exists, its IP is publicly available by users.

9) Database replication can be used in many database management systems, usually with a master/slave relationship between the original (master) and the copies (slaves). All data-modifying operations(insert, update, delete) must be sent to master(s). Read operations are sent to slave machines.

![](./images/001.png)

10) Improving the load/response time can be done by adding a cache layer and shifting static content (JavaScript/CSS/image/video files) to the content delivery network (CDN).

11) A cache is a temporary storage area of expensive operations. Hereby, less db operations are called. Couchbase can be considered as cache in companies.

12) Cache is bringing a solution to solve the problem of lots of DB calls. Cache is located between app and db.

![](./images/002.png)

13) Interacting with cache servers is simple because most cache servers provide APIs for common programming languages. The following code snippet shows typical Memcached APIs:

```java

SECONDS = 1
cache.set('myKey, 'hi there', 3600 * SECONDS)
cache.get('myKey')

```

14) Cache should be used in the scenario where READ operations outnumber MODIFY(INSERT/DELETE/UPDATE) operations. Cache server stores data in memory, therefore this is not appropriate for important data. Thus, important data should be stored in persistent locations such as DB, S3 etc.

15) Some considerations on cache usage

    - Decide when to use: Use when READ operations outnumber
    - Expiration policy: Set an appropriate expiration policy. Too small values lead to more DB calls, too big values lead data to become stale.
    - Consistency: Keep the data in store and the data in cache in sync.
    - Mitigating failures: Prefer using a cluster of cache rather than a single cache server.
    - Eviction policy: Removing some elements from cache when it is full. The removal criteria might be least recently used(LRU), least frequently used(LFU) or First in First Out(FIFO).

16) "A CDN is a network of geographically dispersed servers used to deliver static content". Dynamic content caching is a new concept that means caching of HTML pages but it isn't in the scope of this course. If a CDN server locates in Germany, a user from France accesses to it faster than a user from US. CDN Server has a TTL(time to live) for static contents on itself. If it exceeds the limit, the cached static content will be freed out and will be pulled again from web server or S3.

![](./images/003.png)

17) Stateless apps are preferrable over stateful apps. Staless apps aren't sharing the storage(RDMS, NoSQL). Staless apps are easily scaled. Stateful app can be thought as 2 web servers each having separate DB's and users whose name start with A-B-C-D-E are directed to server 1 and users whose name start with F-G-H-J-I are directed to server 2. The data on server 1 and server 2 are different and the direction happens in load balancer. Whereas, the DB is the same among 2 servers in the scenario of stateless apps.

![](./images/004.png)

18) When there are 2 data centers, users are geoDNS-routed, also known as geo-routed, to the closest data center normally. A challenge of multiple data centers is data syncranization. "In failover cases, traffic might be routed to a data center where data is unavailable".

19) Message Queue is stored in RAM and supports and asynchronous communication. Message queues help decoupling and makes our software loosely coupled.

20) Logging, Metrics and Automation become necessary when things become complicated

21) AWS RDS can provide a database whose ram is 24 TB.

22) Horizontal scaling is also known as sharding. "Sharding separates large databases into smaller, more easily managed parts called shards. Each shard shares the same schema, though the actual data on each shard is unique to the shard". When data wants to be accessed, a hash function  is called. When there are 4 shards, a hash function of mod 4 is called to determine where to allocate/access from where.

![](./images/005.png)

![](./images/006.png)

23) "The most important factor to consider when implementing a sharding strategy is the choice of the sharding key". "Sharding key (known as a partition key) consists of one or more columns that determine how data is distributed". "When choosing a sharding key, one of the most important criteria is to choose a key that can evenly distributed data".

24) Some problems of sharding

    - Resharding data
    - Celebrity problem
    - Join and de-normalization: When DB is sharded, it is hard to perform join operations

25) Denormalization is a way to solve the problem of running complex joins multiple time. When we want to run a complex join query, running it a lot might be costly. In this scenario, what is required is to run the query once and store it on a table. This is known as denormalization. It leads to redundant data on all shards but prevents us from running the same query multiple times.

26) It is a good practice to move non-relational functionalities to a NOSQL DB.

![](./images/007.png)



