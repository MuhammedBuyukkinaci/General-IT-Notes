# Vs code settings

1 ) Global settings for Vscode setup. It is located on /home/muhammed/.config/Code/User/settings.json.

```settings.json
{
    "terminal.integrated.defaultProfile.linux": "zsh",
    "terminal.integrated.profiles.linux": {
        "bash": {
            "path": "bash",
            "icon": "terminal-bash"
        },
        "zsh": {
            "path": "zsh"
        },
        "fish": {
            "path": "fish"
        },
        "tmux": {
            "path": "tmux",
            "icon": "terminal-tmux"
        },
        "pwsh": {
            "path": "pwsh",
            "icon": "terminal-powershell"
        },
        "zsh (2)": {
            "path": "/usr/bin/zsh"
        }
    },
    "window.zoomLevel": 2,
    "workbench.startupEditor": "none",
    "workbench.iconTheme": "ayu",
    "workbench.colorTheme": "Ayu Dark",
    "workbench.settings.editor": "json",
    "workbench.settings.openDefaultSettings": true,
    "workbench.settings.useSplitJSON": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    },
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "code-runner.executorMap": {
        "python": "$pythonPath -u $fullFileName"
    },
    "code-runner.showExecutionMessage": false,
    "code-runner.clearPreviousOutput": true,
    "python.testing.pytestArgs": [
        "pytest"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.languageServer": "Pylance",
}

```

) 
2) Install the following extensions on Vscode marketplace.

  - Python(installing Pylance, pylint(remove it later), jupyter notebook, debugging etc)
  - isort(sorting our imports)
  - Black Formatter to format our files
  - Code Runner to run our code via right-upper green triangle
  - Ayu to have a different theme workplace

# PyCharm Settings

1) Install Rainbow brackets, rainbow csv, rainbow indent as plugins.
2) Download black pycharm formatter and install it via **load from disc**. Install the library
   via `pip install 'black[d]'`. Follow the
   instructions [here](https://black.readthedocs.io/en/stable/integrations/editors.html)
3) Open up plugin pages and type pylint to search. Install and enable the plugin.
4) Open up plugin pages and type **One Dark Theme**. Install and enable the plugin. Choose *One Dark Vivid Italic* as
   color scheme.
5) Create a .pylintrc file in root folder.

# My-Nginx-Notes

1) Http = Hyper Text Transfer Protocol

2) Protocol = A set of rules that are agreed upon 2 or more parties for communication

3) Http uses port 80 of computer/server but it is easily changeable.

4) All modern Web browsers have Network tab to display Network activities.

5) Generally, data transferred on top of Http protocol are zipped via some algorithm provided by web servers and web browsers are able to unzip it in the background.

6) Daemon in unix-like OS = Service in Windows.

7) Some web servers =  Apache, Nginx, Microsoft IIS, Httpd. Httpd is used with PHP generally.

8) Nginx is
  - A http and reverse proxy server
  - A mail proxy server
  - A generic TCP/UDP proxy server

9) According to [Netcraft](https://www.netcraft.com/), a statistic-oriented website, Nginx usage is around 28% as of 2017 February.

10) A statistic for user wait times is blow:
![Statistic](https://github.com/MuhammedBuyukkinaci/General-IT-Notes/blob/master/nginx_images/01_wait_time.png)

11) Why to use Nginx:
  - Slow
  - Acceleration
  - Load balancing
  - Scalable concurrent connections handling
  - The ability to operate on relatively cheap hardware.
  - On the fly upgrades
  - Ease of installation and maintenance


12) In apache, a thread is created for each new request. If you have 1000 request at the same time, you have 1000 threads runnig concurrently. However, in NGINX, one thread is used for 1000 requests but it is based on events. vent pased processing is much much faster.

13) NGINX not only supports HTTP and HTTPS, but also IMAP(Internet Message Access Protocol), POP3 (Post Office Protocol) and SMTP(Simple Mail Transfer Protocol).

14) Apache and NGINX can be used together.

15) Web root directory in Apache is `/var/www/html`.

16) Log files of ngin is under `/var/log/nginx` . You will find access.log and error.log.

17) The binary that starts the nginx daemon is under `/usr/bin` . 

18) The web directory from which nginx serves web pages by default is `/usr/share/nginx`.

19) To start nginx service 

```
sudo service nginx start
```

20) Nginx should be compiled from the source rather than installing via apt if you want to customize its settings. For each option(from source or from ubuntu/debian repositories) configuration files location are below:

  - From Source:
    - /etc/nginx/conf.d/
  - From Repo(symblinks to /etc/nginx/sites-available/):
    - /etc/nginx/sites-enabled/

21) Nginx is utilizing an asynchronous event-driven model, delivering reliable performance under significant loads.

22) Main configuration file is `/etc/nginx/nginx.conf`.

23) In NGINX, configuration options are known as *directives*. These *direvtives* are either in a block or not. Some directives which are listed in the beginning of */etc/nginx/nginx.conf* file are user, worker_processes, error_log, pid etc.

24) **http** and **events** are some blocks in */etc/nginx/nginx.conf*.

25) **http** block encapsulates directives for web traffic handling.

26) *include* directive under **http** block is where to look for configuration files if you are using separate configuration files in /etc/nginx/conf.d/ or /etc/nginx/sites-enabled/

27) **server** block should be under **http** . An example server block is below:

```
server {

listen 80 default_server;

listen [::]:80 default_server;

server_name example.com www.example.com;

root /var/www/example.com;

index index.html;

try_files $uri /index.html;

}
```

28) **location** blocks are located under **server** blocks. "NGINX’s location setting helps you set up the way in which NGINX responds to requests for resources inside the server. As the server_name directive informs NGINX how it should process requests for the domain, location directives apply to requests for certain folders and files (e.g. http://example.com/blog/.) "(from [here](https://www.plesk.com/blog/various/nginx-configuration-guide/#:~:text=Every%20NGINX%20configuration%20file%20will,interchangeably%20as%20blocks%20or%20contexts%20.))

29) An example of **server** and **location** blocks are below:

```
server {

  location / { }

  location /photos/ { }

  location /blog/ { }

  location /home/ { }

  location /blog/categories/ { }

}
```

30) We can use regular expression by writing `location ~ SOME_RE_COMMAND_HERE` in location blocks. For case insensitive, use `~*`. For particular string, use `^~`. To stop searching after finding an exact match, use `=`. Exact match will speed up the performance.
```
# a case sensitive Regex
location ~ ^/myblog(/|/index\.py)$ { }
# case insensitive
location ~* ^/myblog(/|/index\.py)$ { }
# particular string
location ^~ /blog/BlogPlanet/ { }
# Exact match, No search after matching
location = / { }
```

31) *root* directive under location block is telling us where to look to give a response to a request. When "NGINX has identified the location directive that is the best match for a specific request, its response will be based on the associated location directive block’s contents"(from [here](https://www.plesk.com/blog/various/nginx-configuration-guide/#:~:text=Every%20NGINX%20configuration%20file%20will,interchangeably%20as%20blocks%20or%20contexts%20.)). Exact paths can be used for *root* direvtive like /home/myuser/.

```
location / {

root html;# /etc/nginx/html

index index.html index.htm;

}
```

32) The content of index.html and index.htm will be returned if no match situation occurs. *index* directive is used not to show a 404 not found error message to clients.

# MLFlow

[Link 1](https://www.youtube.com/watch?v=859OxXrt_TI&ab_channel=InfoQ)

[Link 2](https://www.youtube.com/watch?v=6z0_n8kxh-g&ab_channel=PyData)

1) MLFlow is a platform for the complete ML lifecycle. It is open source and started at DataBricks. It may be installed locally or on a remote server to be used via multiple developers. MLFlow is language agnostic.

2) ML lifecycle is below:

![ml lifecycle](./mlflow_images/001.jpeg)

3) There are some platforms like Facebook's FBLearner and Uber's Michelangelo and Google's TFX to standardize ML lifecycle. They aren't completely open source. They are limited to a few algorithms or frameworks. [Weights & Biases](https://wandb.ai/site) is paid service like MLFlow, which is a competitor to MLFlow.

4) Characteristis of MLFlow

![characteristics](./mlflow_images/002.jpeg)

5) MLFlow faciliates reproducibility.

6) MLFlow components:

![mlflow_concepts](./mlflow_images/003.jpeg)

7) **Tracking** is a centralized repository for metadata about training sessions within an organization. **Projects** is a reproducible & self contains packaging format for model training code, ensuring that training code runs the same way regardless of execution environment. **Models** is a general purpose model format enabling any model produced with MLFlow to be deployed to a variety of production environments.

8) Some key concepts in Tracking

![tracking](./mlflow_images/004.png)

9) MLFlow Tracking Server

![tracking](./mlflow_images/005.png)

10) MLFlow Tracking usage example

![tracking](./mlflow_images/006.png)

11) A tracking result from MLFlow GUI

![tracking](./mlflow_images/007.png)

12) MLFlow stores entity(metadata) store and artifact store.

![tracking](./mlflow_images/008.png)

13) MLFlow Project is a directory structure. It automates fetching a project from github and run it via parameters defined conda.yaml.

14) Sample MLFlow Project:

![tracking](./mlflow_images/009.png)

15) ML models can be productionized via various solutions. The motivation behing **MLFlow models** is below. The solution to this complexity is a unified model abstraction called **MLFlow model**.

![tracking](./mlflow_images/010.png)

16) A Sample MLFlow models 

![tracking](./mlflow_images/011.png)

17) MLFlow Flavors Example

![tracking](./mlflow_images/012.png)

18) 2 key concepts of MLFlow:

- Run: A collection of hyperparameters, training metrics, labels, and artifacts related to a machine learning model

- Experiment: COnsists of lots of different runs.



# Basic Concepts

1) Web is a service running on the ineternet.  Internet is anarchitecture, web is a service. There are other services like IRC (internet relay chat) , e-mail, Voip and telnet, which run on the internet.

2) Client makes a request to Server via browser.

3) Web is a term used in order to mean servers that use HTTP

4) Client makes a request using HTTP and TCP/IP and server responds to the client.

5) URL = access_method://server_name:port/location/to/go

6) DNS is abbreviating Domain Name Server. It matches domain names and correspondent IP.

7) HTTP is an application protocol.

8) ISP is abbreviating Internet Service Provider like Turkcell Superonline and Türk Telekom.

9) `ping website.com` is a command to check whether a server is up or not.

10) Protocol Stack is built on OS. Protocol stack is referred as TCP/IP protocol stack.

![protocol stack](./network_images/000.png)

11) Data is composed of chunks(packets).

12) On TCP, each packet was given a port number.

13) On IP, each packet was given its destination IP.

14) On Hardware Layer, alphabetic text was converted to eloctronic signal.

15) The packets transmitted over te internet. When packets reach the destination, IP address and port number info was removed because it already reached.

16) `traceroute` is a Unix-like or Windows command to track packages sent to destination

17) Router directs dispatched packets to its destination

18) ROuter examines destination address of packets and determines where to send it.

19) Client Computer -> Modem -> Public Telephone Network -> Modem Pool -> ISP Port Number -> Router(x10) -> ISP Backbone(x10)

20) DNS is a distributed database which keeps track of computer's names and their corresponing IP addresses. My request to a destination first goes to a DNS Server. After obtaining correspondent IP, my computer connects to target computer.

21) The government is able to block youtube.com because ISP's have DNS's. Google is providing a free DNS.

22) HTTP is Hyper Text Transfer Protocol. It is an application protocol that makes the web works.

23) HTTP is the protocol that web browsers and web servers use to communicate with each other over the internet. HTTP sits on top of TCP layer.

24) Ports can be thought of as separate channels on each computers.

25) TCP receives data for application layer protocol and segments it into chunks. TCP assigns port numbers to these chunks.

26) Some important port numbers

| Port    | Operation |
| --------- | ----------- |
|  20,21    | FTP         |
|  22       | SSH, SFTP   |
|  23       | Telnet      |
|  53       | DNS         |
|  80       | HTTP        |
|  443      | HTTPS       |
|  3389     | RDP         |

27) IP's job is to send and route packets to other computers

28) Packet has IP Header, TCP Header, data from application layer.

29) FTP Server: a computer on the internet that offers FTP

30) Filezilla is an FTP client. There are other methods like using Terminal or using browser.

31) There are 2 types of IP:

- Public IP: Same for devices connected to same modem

- Private IP: Assigned by modem. For each device, it is unique.

32) Public IP address may change around time. A private IP address might be like 192.168.1.100 etc.

33) Routers are handling internet traffic for multiple computers at once using a single public IP address.

34) There are 65525 TCP ports in a computer.

35) Modern internet more closely follows the simpler internet protocol suite, known as TCP/IP.

36) Search engine is a web service helping us find other web pages.

37) Static web server is a server that sends its hosted files to browser.

38) Dynamic web server is a static web server + application server + database.

39) A protocol is a set of rules for communication between 2 computers.

40) Sending an HTML form from our browser to server is made using HTTP.

# FastAPI Notes

1) FastAPI is very fast to make an API.

2) Traditionally, we aren't explicitly defining what kind of data our API is expecting(Flask, Django). When you create an API in FastAPI, you define what kind of data your API is expecting. First advantage of FastAPI is data validation.

3) Another advantage of FastAPI is automatic documentation. It is using FastAPI.

4) The last advantage of FastAPI is AutoCompletion and Code Suggestions in VS Code and PyCharm.

5) To use FastAPI, install FastAPI and Uvicorn via `pip install fastapi` and `pip install uvicorn`. Uvicorn is enabling us to run our API's like a web server.

6) An API Endpoint(can be also called as Route or Path) is the point of entry in a communication channel when two systems are interacting. It refers to touchpoints of the communication between an API and a server. It is like /hello or /get-item.

7) Endpoint HTTP Verbs:

- When we have an endpoint that has a GET method, this endpoint will be returning information.       
- When we have an endpoint that has a POST method, this endpoint will be creating something new(adding a record to DB etc). 
- When we have an endpoint that has a PUT method, PUT method updates the information.
- When we have an endpoint that has a DELETE method, DELETE method deletes the information.
- There are other HTTP words which aren't used frequenly as above one. These are OPTIONS, HEAD, PATCH, TRACE.

8) To run uvicorn, run the following. temp is our file as temp.py but .py not typed. Click [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). An alternative documentation is working on [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

```uvicorn.sh
uvicorn temp:app --reload
```

```first_api.py
from fastapi import FastAPI

# Creating a FastAPI instance
app = FastAPI()

@app.get("/")
def home():
    return {"Data": "Testing"}

```

9) Any response coming from an endpoint will be converted to JSON by FastAPI in the background.

10) We can send path parameters to /get-item/{item_id} endpoint via [http://127.0.0.1:8000/get-item/1](http://127.0.0.1:8000/get-item/1) and get ***{"name":"bread","price":5.99}***. [http://127.0.0.1:8000/get-item/4](http://127.0.0.1:8000/get-item/4) will return ***Internal Server Error***. [http://127.0.0.1:8000/get-item/any_object](http://127.0.0.1:8000/get-item/any_object)
will return ***{"detail":[{"loc":["path","item_id"],"msg":"value is not a valid integer","type":"type_error.integer"}]}***.

11) We can pass 2 path parameters {item_id, name} to an endpoint like we did in *get_item_2_path_params* function.

12) `from fastapi import Path` is allowing us to add more details or enforcements or constraints to our actual path parameter. Before the information, we need to set a default value to Path. gt means greater than, lt means less than in Path parameters.

13) Query parameter is something that comes after a question mark(?) in a url. An example is https://example.com?redirect=/home&message=fail. **get_item_by_query** is implementing query parameter example. An example url is [http://127.0.0.1:8000/get-by-name?name=milk](http://127.0.0.1:8000/get-by-name?name=milk) and it returns *{"name":"milk","price":20.99}*. [http://127.0.0.1:8000/get-by-name?name=banana](http://127.0.0.1:8000/get-by-name?name=banana) returns *{"Data":"Not Found"}*. We have to fill query parameter if we want to send a request to relevant url. If we don't want to make it obligatory, a default None value should be passed to query parameter. We can define some query parameters as required, some as having a default value, and some entirely optional. product_id is a path parameter. brand is an obligatory query parameter. size is a query parameter having a default value and not have to be filled out. Color is a query parameter which can be None or str but it is None by default.

```query.py
from typing import Union
@app.get('get-product/{product_id}'):
def get_proruct(product_id: int, brand: str, size: str = 'L', color: Union[str, None] = None ):
    pass

```

14) We can combine path parameters and query parameters in an endpoint. *get_item_by_path_param_query_param* does exatly this. [http://127.0.0.1:8000/get-by-name-path-param-and-query/3?test=2](http://127.0.0.1:8000/get-by-name-path-param-and-query/3?test=2) will run **get_item_by_path_param_query_param** function and return *{"name":"egg","price":3.0}*.

15) *create_item* function is a post function. It adds a new record to database(python dictionary in our case). The body should cover the attributes of Item class. Item class is inherited from Basemodel of PyDantic. Open up [docs](http://127.0.0.1:8000/docs) page and send a body via POST /create-item/{item_id}.

16) update_item function is updating an existing data. It is a PUT request. It uses UpdateItem pydantic class.

17) delete_item function is deleting an existing data. If data doesn't exist, it returns a 404 status code.

18) To try POST, PUT and DELETE, run POST request first. Run PUT request second. Run DELETE request last.

19) Typer is a CLI of FastAPI.

20) `from fastapi import FastAPI` is a class that inherits directly from [Starlette](https://www.starlette.io/). We can use the existing features of Starlette.

21) The functions used by FastAPI decorators can return dict, list, int, str, Pydantic Models etc. They are going to be converted into json in the background by FastAPI.

22) We don't have to specify the data type in a function listening an endpoint. However,  we will lose the advantages of having a type hint in this scenario.

23) The order in the file having functions that are decorated by FastAPI matter. Below is an example. We should define `/users/me` before `/users/{item_id}`. This is because `/users/me` also fits `/users_user_id`.

```order_matters.py
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

24) We can't have the same endpoint for 2 functions.

25) If we want to have path parameters but we want these parameters to be predefined, we should use FastAPI with Enum as below.

```predefined_enum.py
from enum import Enum


class Sport(str, Enum):
    football = "football"
    basketball = "basketball"


@app.get('/get-sport/{sport}')
def get_sport(sport: Sport):
    if sport == Sport.basketball:
        return {"Sport": "basketball"}
    # another way of accessing value
    if sport.value == "football":
        return {"Sport": "football"} 
    return {"Message": "Sport not found in Enum class"}

```

26) [OpenAPI](https://github.com/OAI/OpenAPI-Specification) is a standard to define API's. Following OpenAPI instructions is the best practice.

27) We can have path of a file in endpoint having path parameter. For more details, check out [this link](https://fastapi.tiangolo.com/tutorial/path-params/#path-parameters-containing-paths).

28) A request body is data sent from our client(browser etc.). Response body is data sent from our API. We can generally send request body via POST, PUT and DELETE. However, we can send data via GET but it isn't best practice. The recommended way to send data is to use PyDantic.

29) PyDantic is similar to dataclass. However, it is more validation oriented. PyDantic has a Pycharm plugin in Pycharm. We can convert pydantic objects to dictionaries.

30) We can use request body parameters, path parameters and query parameters at the same time like below.

```all_here.py
@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Union[str, None] = None):
    result = {"item_id": item_id, **item.dict()}
```

31) `from fastapi import Query` is a way to add extra validation for query parameters. We can add more validation like min_length, regular expression etc. We can pass default values into Query class instead of barely setting equal. We can add title and description to Query class to be displayed on documentation. Also, we can use alias parameter of Query to resolve query parameter that aren't suitable for Python. A query parameter named my-string isn't a valid string name in Python. alias parameter is assigned to Query class and it will map to q in function decoreated by @app.get.

```query.py
async def read_items(q: Union[str, None] = Query(default=None, max_length=50)):

```

32) We can pass many query parameters with same query parameter. [http://127.0.0.1:8000/get-multiple-query/?q=Muhammed&q=Ali](http://127.0.0.1:8000/get-multiple-query/?q=Muhammed&q=Ali) will return **["Muhammed","Ali"]** . We can also assign default values to multiple query parameters via Query class. An example is **['Hasan', 'Hüseyin']**.

```
@app.get('/get-multiple-query/')
def get_multiple_query(q: Union[List[str], None] = Query(default=None)):
    return q
```

# General-IT-Notes
Including my experiences on Software Development

1) Use always explicit directory in crontab. Don't use the former, use the latter.

```crontab -e
# Don't use
* * * * * bash runall.sh
# Use
bash /path/to/directory/runall.sh
```

2) Don't FTP python environment(conda or venv etc.) to VPS. Instead, upload requirements.txt and install it via 

```
source ENVIRONMENT_NAME/bin/activate

pip install -r requirements.txt
```

3) If a python loop is slow, consider to replace list with set. Set has no indexes therefore it is faster especialy checking an element in an array-like (list, tuple, set) object.

4) htop is a good monitoring tool on Unix alike OS's.

5) On Mac, default shell is zsh, not bash.

6) Asbru Connection manager is a good & free ssh client. Available on Linux. Similar to Mobaxterm, SecureCRT of Mac and Windows.

7) When the electricity powers off or the power cable is broken, install GRUB bootloader menu via inserting an ubuntu medium in a live session if the boot menu doesn't show up.

8) Kowl is a Web GUI to display Apache Kafka Topics.

9) [Jinja Templating](https://jinja.palletsprojects.com/en/3.1.x/) is a fast, expressive, extensible templating engine in Python. We are using it on Django Templates

10) Google Data Studio is completely free visualization tool from Google.

11) Google BigQuery is a big data analytics product from Google. Its architecture is serverless.

12) [Unsplash](https://unsplash.com/) is a websie having lots of high-quality pictures.




