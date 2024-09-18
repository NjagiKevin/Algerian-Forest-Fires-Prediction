from pydantic import BaseModel


class UserInput(BaseModel):
    temperature:float
    Ws:float
    Rain:float
    FFMC:float
    DMC:float
    ISI:float
    FWI:float
    classes:str
    region:str


