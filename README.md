# hospitanama-bot
An AI chatbot that helps users gain information about hospitals in Lahore. 

## INSPIRATION
Due to a lack of centralized platform that contains publicly available data about hospitals, Pakistanis primarily rely on word-of-mouth to find the right hospital to go to, or may avoid seeking healthcare entirely. This compounds the stress and adds to the confusion that patients experience, as well as delay timely healthcare delivery. This problem is experienced particularly by those with limited access to the internet, especially low to middle income families, but can be felt by individuals across Pakistan. It is most acute during medical emergencies, where diagnostic delays can directly affect survival; however, patients experience this routinely when seeking healthcare. Existing public healthcare platforms are built primarily for internal administration or strictly medical related purposes; most patient-facing platforms focus on specialist books (such as Marham or Oladoc), not on hospital level facilities and with little to no focus on disclosing hospital quality or any comparison based tool. These platforms are also difficult to use and do not make information digestible for the average person.

This project aims to solve this problem by leveraging artificial intelligence deliver easy to understand information about 45 hospitals in Lahore, as a small proof of concept project to show the benefits of a patient-facing centralized healthcare platform. Instead of digging through a confusing UI, having to comb across multiple different websites and calling hospitals themselves, users can simply talk to ElajFinder (literally: remedy finder!).

## WHAT IT DOES
Uses a novel dataset I created myself; Lahore currently has no publicly available dataset collecting statistics for both private and public hospitals. I made a list of 45 general hospitals in Lahore with over a hundred Google Reviews. For each, I collected various data points by synthesizing information from multiple sources, including public databases and official hospital websites. This dataset also includes negative Google Reviews (1-3 stars) of each hospital, grouped into themes of complaints across six major categories (Environment, Behavior, Management, Money, Wait Time and Department Specific complaints) using BERTopic AI.

Can understand your query even if it's misspelled, abbreviated, or referred to casually. I wanted to ensure that the chatbot could recognize hospital names as they are referred to in common speech (for example: IMC instead of Integrated Medical Complex). In order to make the chatbot accessible to a wide variety of users and recognizing that the vast majority of Pakistanis feel more comfortable communicating in Urdu, it can also respond to users in Roman Urdu and Nastaliq Urdu.

Makes asking questions easier. If the user doesn't want to type in a query, I provide a sidebar where the user can customize their own commonly asked questions. Users can select a hospital from a dropdown to learn more about it, compare two hospitals, or select the features they are looking for a hospital. This automatically generates a personalized query quickly that the bot can answer.

Every answer is grounded in review data. I didn't want ElajFinder to just list statistics at the user; instead, it calculates the ratio of negative reviews as a percentage of the whole, the frequency of certain themes, and uses information from each theme to help give an overview of patient experience at the hospital. It also creates benchmarks against a city-level to ensure that the user gets a look at how the hospital compares to others. It explains each in a conversational, easy to understand manner.

Factors in your location when answering your questions. When the user enters their location into the sidebar, ElajBot automatically calculates drive times and approximate distance to the hospital, for better and more personalized recommendations on which hospital is best suited for the user's needs.

Can handle follow up questions. ElajBot automatically remembers the context from the user's messages, so conversational flow remains natural and just like talking to a friend.

## USE
Link here --> https://elajfinder.streamlit.app/
