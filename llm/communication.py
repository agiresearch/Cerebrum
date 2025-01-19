class Response:
    def __init__(self, 
                 response_message=None,
                 tool_calls=None,
                 finished=False,
                 error=None,
                 status_code=200):
        self.response_message = response_message
        self.tool_calls = tool_calls
        self.finished = finished
        self.error = error
        self.status_code = status_code 