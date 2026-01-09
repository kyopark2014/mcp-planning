import logging
import sys
import boto3
import time
import os
import re
import uuid
import base64
from io import BytesIO
from urllib import parse
import json

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("agentcore-coder")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")
    
def load_config():
    config = None
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

config = load_config()

region = config.get('region', 'us-west-2')
projectName = config.get('projectName')

s3_prefix = 'docs'
s3_image_prefix = 'images'
model_name = "Claude 4.5 Haiku"
s3_bucket = config.get('s3_bucket')
path = config.get('sharing_url')
code_interpreter_id = config.get('code_interpreter_id', 'aws.codeinterpreter.v1')

client = boto3.client(
    "bedrock-agentcore", 
    region_name=region,
    endpoint_url=f"https://bedrock-agentcore.{region}.amazonaws.com"
)

sessionId = None
def create_code_interpreter_sessionId():
    session_id = None
    response = client.list_code_interpreter_sessions(
        codeInterpreterIdentifier=code_interpreter_id,
        maxResults=5,
        status='READY'
    )
    items = response['items']

    if items is not None:
        for item in items:
            session_id = item['sessionId']
            break
    
    if session_id is None:  # still no sessionId
        logger.info("No ready sessions found")
        response = client.start_code_interpreter_session(
            codeInterpreterIdentifier=code_interpreter_id,
            name="agentcore-code-session",
            sessionTimeoutSeconds=900
        )
        logger.info(f"response of start_code_interpreter_session: {response}")
        session_id = response['sessionId']

    return session_id

def get_code_interpreter_sessionId():
    global sessionId
    if sessionId is None:
        sessionId = create_code_interpreter_sessionId()
        logger.info(f"sessionId: {sessionId}")
    else:
        logger.info(f"sessionId: {sessionId}")
        try:
            response = client.get_code_interpreter_session(
                codeInterpreterIdentifier=code_interpreter_id,
                sessionId=sessionId
            )
            logger.info(f"response of get_code_interpreter_session: {response}")        

            status = response['status']
            logger.info(f"status: {status}")
            if status != 'READY':
                logger.info(f"sessionId: {sessionId} is not ready")
                sessionId = create_code_interpreter_sessionId()
                time.sleep(5)
        except Exception as e:
            logger.info(f"error of get_code_interpreter_session: {e}")
            sessionId = create_code_interpreter_sessionId()

    return sessionId

def agentcore_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    
    # get the sessionId
    sessionId = get_code_interpreter_sessionId()
    
    execute_response = client.invoke_code_interpreter(
        codeInterpreterIdentifier=code_interpreter_id,
        sessionId=sessionId,
        name="executeCode",
        arguments={
            "language": "python",
            "code": code
        }
    )
    logger.info(f"execute_response: {execute_response}")

    # Extract and print the text output from the stream
    result_text = ""
    for event in execute_response['stream']:
        if 'result' in event:
            result = event['result']
            if 'content' in result:
                for content_item in result['content']:
                    if content_item['type'] == 'text':
                        result_text = content_item['text']
                        logger.info(f"result: {result_text}")

    # stop the session
    # client.stop_code_interpreter_session(
    #     codeInterpreterIdentifier=code_interpreter_id,
    #     sessionId=sessionId
    # )
    return result_text

def get_contents_type(file_name):
    if file_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif file_name.lower().endswith((".pdf")):
        content_type = "application/pdf"
    elif file_name.lower().endswith((".txt")):
        content_type = "text/plain"
    elif file_name.lower().endswith((".csv")):
        content_type = "text/csv"
    elif file_name.lower().endswith((".ppt", ".pptx")):
        content_type = "application/vnd.ms-powerpoint"
    elif file_name.lower().endswith((".doc", ".docx")):
        content_type = "application/msword"
    elif file_name.lower().endswith((".xls")):
        content_type = "application/vnd.ms-excel"
    elif file_name.lower().endswith((".py")):
        content_type = "text/x-python"
    elif file_name.lower().endswith((".js")):
        content_type = "application/javascript"
    elif file_name.lower().endswith((".md")):
        content_type = "text/markdown"
    elif file_name.lower().endswith((".png")):
        content_type = "image/png"
    else:
        content_type = "no info"    
    return content_type

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
        )

        content_type = get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        #url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        url = path+'/'+s3_image_prefix+'/'+parse.quote(file_name)
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def generate_short_uuid(length=8):
    full_uuid = uuid.uuid4().hex
    return full_uuid[:length]

def agentcore_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    Do not save pictures locally bacause the runtime does not have filesystem.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    code = re.sub(r"plt.show", "#plt.show", code)

    post = """\n
import io
import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()

print(image_base64)
"""
    code = code + post    
    logger.info(f"code: {code}")

    # get the sessionId
    sessionId = get_code_interpreter_sessionId()
    
    execute_response = client.invoke_code_interpreter(
        codeInterpreterIdentifier=code_interpreter_id,
        sessionId=sessionId,
        name="executeCode",
        arguments={
            "language": "python",
            "code": code
        }
    )
    logger.info(f"execute_response: {execute_response}")

    # Extract and print the text output from the stream
    result_text = ""
    for event in execute_response['stream']:
        if 'result' in event:
            result = event['result']
            if 'content' in result:
                for content_item in result['content']:
                    if content_item['type'] == 'text':
                        result_text += content_item['text']
                        logger.info(f"result chunk: {content_item['text']}")
    
    logger.info(f"full result_text length: {len(result_text)}")
    
    # Clean and fix base64 string
    base64Img = result_text.strip()
    # Remove any whitespace and newlines
    base64Img = ''.join(base64Img.split())
    
    image_url = None
    
    if base64Img:
        # Fix padding if needed
        missing_padding = len(base64Img) % 4
        if missing_padding:
            base64Img += '=' * (4 - missing_padding)
        
        try:
            byteImage = BytesIO(base64.b64decode(base64Img))
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            logger.error(f"base64Img length: {len(base64Img)}, first 100 chars: {base64Img[:100]}")
            raise

        image_name = generate_short_uuid()+'.png'

        os.makedirs('contents', exist_ok=True)
        file_path = os.path.join('contents', image_name)
        with open(file_path, 'wb') as f:
            f.write(byteImage.getvalue())

        image_url = upload_to_s3(byteImage.getvalue(), image_name)
        logger.info(f"image_url: {image_url}")
    else:
        logger.error("No base64 image data received")
        raise ValueError("No base64 image data received from code interpreter")

    return {
        "path": image_url
    }
