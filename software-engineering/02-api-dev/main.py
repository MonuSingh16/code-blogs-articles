from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel

app = FastAPI()


"""
# Add all notes/comments here

"@"" -> A Decorator. "get" -> HTTP Method for retrieving data as request. "/"" -> Path Operation
In case of same path operation, then the first match path outputs. Hence, Order matters.

"""

class Post(BaseModel):
    title: str
    content: str
    body: str

@app.get("/")
async def root():
    return {"message": "Welcome to my API"}

@app.get("/posts")
def get_posts():
    return  {"data": "This is your post"}

@app.post("/createposts")
def create_posts(payLoad: dict = Body(...)):
    print(payLoad)
    return {"new_post": f"title: {payLoad['title']} content: {payLoad['content']}"}

@app.post("/createnewposts")
def create_posts(new_post: Post):
    print(new_post.title)
    return {"data": "new post"}

"""
You have not work for a while
You have not work for a while

"""