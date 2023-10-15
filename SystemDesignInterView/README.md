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

# Back-of-the-envelope Estimation

1) Back-of-the-envelope calculation means rough/approximate calculation.

2) A byte is a sequence of 8 bits. An ASCII character uses one byte of memory (8 bits).

![](./images/008.png)

3) 1 ms(millisecond) = 10^-3 seconds = 1,000 µs(microsecond) = 1,000,000 ns(nanosecond)

4) Some latency numbers

![](./images/009.png)

![](./images/010.png)

5) Avoid disc seeks and prefer using RAM. Compress data before sending over the network.

6) QPS means query per second. 

7) Commonly asked back-of-the-envelope estimations: QPS, peak QPS, storage, cache, number of servers.

# A Framework For System Design Interviews

1) Asking good questions is vital in system design interviews.

2) DAU means daily active user.

# Design A Rate Limiter

1) "In a network system, a rate limiter is used to control the rate of traffic sent by a client or a service". It can be time-based, IP-based or device based. Rate limiter blocks excess calls.

2) DoS means Denial of Service. 

3) Python libraries [fastapi-limiter](https://pypi.org/project/fastapi-limiter/) and [slowapi](https://pypi.org/project/slowapi/) are 2 libraries to set a rate limit in FastAPI.

4) Rate limiting can be server-side or client-side. Client-side rate limiting is untrusted because it is vulnarable to misuse of forgers. Rate limiting can be programmed in the application code or as a separate service.

5) Rate limiting can be in the middleware as below:

![](./images/011.png)

6) Rate limiting is implemented in API Gateway in cloud apps. "API gateway is a fully managed service that supports rate limiting, SSL termination, authentication, IP whitelisting, servicing static content". Rate limiting can be implemented in API Gateway or server-side applications(FastAPI etc.). There is no clear answer to this question.

7) Evolution of an API architecture

![](./images/012.png)

8) Algorithms for rate limiting

- Token bucket: Simple, well understood and commonly used by Amazon and Stripe. Bucket size and refill rate are 2 parameters.

![](./images/013.png)

- Leaking bucket: Bucket size and outflow rate are 2 parameters. Shopify uses this algorithm.

![](./images/014.png)


- Fixed window counter:

![](./images/015.png)

- Sliding window log:

![](./images/016.png)

- Sliding window counter:

![](./images/017.png)

9) High level architecture of rate limiting. At the high level, we need a counter of how many requests are sent from the same user, IP address etc.

![](./images/018.png)

10) Redis is a popular option for rate limiting as an in-memory store.

11) Lyft open sourced its rate limiting library as [here](https://github.com/envoyproxy/ratelimit). It works as a configuration file.

12) When a request is rate limited, a HTTP response called 429 is returned.

![](./images/019.png)

13) When a rate limiter server is enough, it is necessary to scale up rate limiter server.

14) A good practice of multiple rate limiters with redis

![](./images/020.png)

15) "Multi-data center setup is crucial for a rate limiter because latency is high for users located far away from the data center"

16) We talked about rate limiting at the application level(layer 7) but it is possible to rate limit at other layers. We can rate-limit IP addresses using IP Tables(IP layer 3).

17) Soft rate limiting is permitting users to send requests for a short period of time even if it exceed rate limit. Hard rate limiting isn't rrmitting users to exceed the threshold.

18) Avoid being rate-limited. Use client cache to avoid frequent API calls. Understand the limit and don't send too many requests in a short period of time.

# Design Consistent Hashing

1) For horizontal scaling, it is significant to distribute requests over servers. Hashing is important in this topic.

2) A basic approach as below. However, it is problematic in the case of adding and removing servers.

![](./images/021.png)

3) "Consistent hashing is a special kind of hashing such that when a hash table is re-sized and consistent hashing is used, only k/n keys need to be remapped on average, where k is the number of keys, and n is the number of slots. In contrast, in most traditional hash tables, a change in the number of array slots causes nearly all keys to be remapped"

4) SHA-1 is a commonly used hash function.

5) A demonstration of hash space and hash keys

![](./images/022.png)

6) Go clockwise from the key position on the ring until a server is found.

![](./images/023.png)

7) Adding a server or removing a server is easy in consistent hashing. However, when a server goes offline, its traffic will go to the clockwise server. In this way, traffic will be less even. 

8) A further solution to above problem is to use virtual nodes. The more the virtual nodes, the less the imbalance of distribution.

![](./images/024.png)

9) Consistent hashing is used in partitioning component of Amazon's DynamoDB, partitioning across the cluster in Apache Cassandra, Discord chat application etc.

# Design A Key-value Store

1) "A key-value store, also referred to as a key-value database, is a non-relational database". Short keys perform better. Key can be plain text like this_is_a key or hashed value like AB241AD56.

2) "An intuitive approach is to store key-value pairs in a hash table, which keeps everything in memory".

3) To overcome the problems of using single server key-value store, we should prefer

    - Data compression
    - Store only recent data in memory and rest in disc.

4) A distributed key-value store is also called a distributed hash table, which distributes data across many servers.

5) CAP Theorem states that it is impossible to provide Consistency, Availability and Partition tolerance at the same time.
    - Consistency: All clients viewing the same data
    - Availability: All clientst having a response even if some nodes are down.
    - Partition Tolerance: A partition indicates a communication break between two nodes. Partition     tolerance means the system continues to operate despite network partitions

![](./images/025.png)

6) Meanings of different combinations

    - CP: Consistency and Partition Tolerance preferred over Availability. Banks prefer. MongoDB follows this combination.
    - AP: Availability and Partition Tolerance preferred over Consistency. Cassandra follows this combination.
    - CA: Consistency and Availability preferred over Partition Tolerance. Since network failure is unavoidable, CA isn't possible in real life.

7) In reality, partitions(breaks) happen in a distributed system.

## Data partition

8) It isn't feasible to store all data in a single server. Thus, it is required to distribute data across many servers. One way to achieve this is to use [consistent hashing](#design-consistent-hashing). "Using consistent hashing to partition data has the following advantages":

- Automatic scaling: Automatically adding and removing servers

- Heterogeneity: The number of virtual nodes is proportional to the capacity of server.

## Data replication

9) Data should be replicated over N servers. N is a tunable parameter. In the scenario below, data is replicated in 3 servers. The logic is to choose 3 servers in the clockwise way. Virtual Nodes shouldn't be taken into consideration.

![](./images/026.png)

## Consistency

10) Data should be syncronized since it is distributed over multiple servers.

11) Some hyperparameters to tune

    - N: The number of replicas
    - W: A write quorum of size W. In order for a write operation to be regarded as successful, write operation must be acknowledged by at least W replicas
    - R: A read quorum of size W. In order for a read operation to be regarded as successful, read operation must be acknowledged by at least R replicas.

12) "The configuration of W, R and N is a typical tradeoff between latency and consistency". If W or R is 1, an operation is returned quickly. If W or R > 1, it offers better consistency but less speed.

    - If R = 1 and W = N, the system is optimized for a fast read.

    - If W = 1 and R = N, the system is optimized for fast write.

    - If W + R > N, strong consistency is guaranteed (Usually N = 3, W = R = 2).

    - If W + R <= N, strong consistency is not guaranteed.

13) A consistency model defines the degree of data consistency.

    - Strong consistency: "Any read operation returns a value corresponding to the result of the most updated write data item. A client never sees out-of-date data".
    - Weak consistency: "Subsequent read operations may not see the most updated value".
    - Eventual consistency: "This is a specific form of weak consistency. Given enough time, all updates are propagated, and all replicas are consistent". Dynamo and Cassandra prefer this.

14) "A [vector clock](https://en.wikipedia.org/wiki/Vector_clock) is a [server, version] pair associated with a data item. It can be used to check if one version precedes, succeeds, or in conflict with others".

## Handling failures

15) In a distributed system, at least 2 servers are required to tell that another server is down.

16) [Gossip protocol](https://en.wikipedia.org/wiki/Gossip_protocol#:~:text=A%20gossip%20protocol%20or%20epidemic,all%20members%20of%20a%20group.) is a way to detect decentralized server detection.

17) Sloppy quorum and hinted handoff are 2 techniques to deal with temporary failures. Sloppy quorum is a technique to choose write/read servers. "The first W healthy servers for writes and first R healthy servers for reads on the hash ring". It is an alternative to quorum requirement described above. Hinted handoff means that server A processes requestes of server B if server B is down. If server B becomes up, the data on B gets updated and then requests are directed to server B.

19) It is significant to replicate data over multiple data centers.

20) Merkle tree is a technique to handle permanent failures.

21) A coordinator is a node that acts as a proxy between the client and the key-value store.

![](./images/027.png)

22) A summary table of above ones

![](./images/028.png)

# Design A Unique ID Generator In Distributed Systems

1) The canonical way to generate unique id's in RDBMS is to use a primary key with auto increment attribute.

2) Auto increment doesn't work in a distributed environment.

3) Unique ID must be unique and sortable.

4) The options to generate unique ID's

- Multi-master replication: Using the auto increment feature of DB's. Instead of increasing one, increase by k.

![](./images/029.png)

- Universally unique identifier (UUID): "UUID is a 128-bit number used to identify information in computer systems". "Here is an example of UUID: 09c93e62-50b4-468d-bf8a-c07e1040bfb2".

- Ticket server: Flickr uses. "The idea is to use a centralized auto_increment feature in a single database server (Ticket Server)"

![](./images/030.png)

- Twitter snowflake approach:

![](./images/031.png)

# Design A URL Shortener

1) Tinyurl is a url shortener service. It is a paid service.

2) Url shortening is a commonly asked system design question.

3) The ratio of read operation over write operation is around 10 on average.

4) A URL Shortener has the following 2 endpoints:

    - URL shortening: POST request. Inputting a long url, converting it to a short url and outputting a short url.

    - URL redirecting: GET request. Inputting a short url and returning a long url.

5) This is what happens when a shortened url is entered into a browser.

![](./images/032.png)

6) There are 2 redirection policies:

    - 301 redirect:

    - 302 redirect:

7) "The most intuitive way to implement URL redirecting is to use hash tables".

8) URL Shortening high level design. It uses a hash function.

![](./images/033.png)

9) In the high-level design, hash table is stored in memory. However, the memory is limited and expensive. Thus, hash table should be stored in a RDBMS.

10) Hash function is a function that is used to convert long url into short url.

11) "The hashValue consists of characters from [0-9, a-z, A-Z], containing 10 + 26 + 26 = 62 possible characters".

12) There are 2 types of hash functions.

- Hash + collision resolution: "The first approach is to collect the first 7 characters of a hash value".

![](./images/034.png)

- Base 62 conversion: Base conversion helps to convert the same number between its different number representation systems

![](./images/035.png)

13) The comparison of Hash + collision resolution and Base 62 conversion

![](./images/036.png)

14) A general flow:

![](./images/037.png)

15) URL Redirecting with cache, db, load balancer and web servers.

![](./images/038.png)

16) "Integrating an analytics solution to the URL shortener could help to answer important questions like how many people click on a link".

# Design A Web Crawler

1) A web crawler is also known as a robot or a spider. Web crawlers are used by search engines.

2) Web crawlers go to a website, then find links on that site, go to links. This lifecycle happens iteratively.

![](./images/039.png)

3) Why to use web crawlers:

    - Search engine indexing: The most comon case

    - Web archiving:

    - Web mining:

    - Web monitoring:

4) The algorithm of web crawlers as follows:

    - Given a set of URLs, download all the web pages addressed by the URLs.

    - Extract URLs from these web pages

    - Add new URLs to the list of URLs to be downloaded. Repeat these 3 steps.


5) An average web size is 500 kilobytes.

5) 29 percent of pages on the web are duplicates.

6) A high-level design of web crawler

![](./images/040.png)

7) Components of a web crawler:

- Seed URL's: Started as starting point of the crawl process. It can be based on domain or country.

- URL Frontier: Storing URL's to be downloaded. Can be considered as First in First Out queue.

- HTML Downloader: Downloading web pages from the internet.

- DNS Resolver: Translating URL into an IP address

- Content Parser: Parsing and validating the downloaded page.

- Content Seen?: Eliminating data redundancy and shortening process time.

- Content Storage: Storing HTML content. Can be in RAM or disc.

- URL Extractor: Parsing and extracting links from HTML pages

- URL Filter: Excluding some URL's such as blacklisted ones, error links etc.

- URL Seen?: Helping to avoid the same URL multiple times.

- URL Storage: Storing already visited URL's






