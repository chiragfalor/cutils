
from openai import OpenAI
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))

with open(f'{cur_dir}/../../openai_utils/openai-api.key', 'r') as f:
    api_key = f.read()

client = OpenAI(api_key=api_key)


def get_model_response(user_prompt: str, *, system_prompt: str | None=None, model: str="gpt-4o-2024-08-06"):

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages += [{"role": "system", "content": system_prompt}]

    messages += [{"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    model_message = response.choices[0].message.content

    return model_message

from pydantic import BaseModel
from datetime import datetime


def structured_response_to_calendar_JSON(response: str):
    class CalendarEvent(BaseModel):
        title: str
        start_time: str
        end_time: str
        location: str
        description: str
        


    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
            {"role": "system", "content": "Extract the event information from the structured response and provide it in a JSON format to be used in a calendar application."},
            {"role": "user", "content": response}
        ],
        response_format=CalendarEvent,
    )
    event = completion.choices[0].message.parsed
    return event.dict()

def email_to_calendar_event(subject: str, email: str):
    system_prompt = """You will be given an email about an event. Read the email carefully and provide a structured response with details of the event. Your response should include: 
    - Title: A concise name for the event
    - Start Time: The date and time the event starts
    - End Time: The date and time the event ends
    - Location: It could be a classroom number or a building name or a venue
    - Description: A very concise one-line description of the event. Don't include filler words. Mention if food will be provided in the same sentence.
    You may include any other details you think are relevant.
    """
    user_prompt = f"Subject: {subject}\n\n{email}"
    response = get_model_response(user_prompt, system_prompt=system_prompt)
    return structured_response_to_calendar_JSON(response)

if __name__ == '__main__':
    subject = "Meeting with the CEO"
    email = """Dear Team,
    We have a meeting scheduled with the CEO on Monday, 1st August 2022 at 2:00 PM. The meeting will be held in the conference room on the 5th floor. Lunch will be provided.
    Please make sure to be on time.
    Regards,
    John"""
    from pprint import pprint
    pprint(email_to_calendar_event(subject, email))
    