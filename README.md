# QuAC-Information-Extraction

### Dataset
Clustering title or keywords in [DRCD](https://github.com/DRCKnowledgeTeam/DRCD), [CMRC](https://github.com/ymcui/cmrc2018) into groups (pseudo-classes).

> "title_label": 8
> -> Passage group #8.

title
dataset


### Dataset Format
```
{
  "title": "Malayali", 
  "type_label": 0,
  "data": [{
    "section_title": "Geographic distribution and population", "background": "The Malayali people or Keralite people (also spelt Malayalee, Malayalam script: mlyaalli and keerlliiy[?]) are an Indian ethnic group originating from the present-day state of Kerala, located in South India. They are identified as native speakers of the Malayalam language, which is classified as part of the Dravidian family of languages. As they primarily live in Kerala, the word Keralite is used as an alternative to Malayali. According to the Indian census of 2011, there are approximately 33 million Malayalis in Kerala, making up 96.7% of the total population of the state.",
    "paragraphs": [{
      "context": "According to the Indian census of 2001, there were 30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India, and 96.7% of the total population of the state. There were a further 701,673 (2.1% of the total number) in Karnataka, 557,705 (1.7%) in Tamil Nadu and 406,358 (1.2%) in Maharashtra. The number of Malayalam speakers in Lakshadweep is 51,100, which is only 0.15% of the total number, but is as much as about 84% of the population of Lakshadweep. In all, Malayalis made up 3.22% of the total Indian population in 2001. Of the total 33,066,392 Malayalam speakers in India in 2001, 33,015,420 spoke the standard dialects, 19,643 spoke the Yerava dialect and 31,329 spoke non-standard regional variations like Eranadan. As per the 1991 census data, 28.85% of all Malayalam speakers in India spoke a second language and 19.64% of the total knew three or more languages.  Large numbers of Malayalis have settled in Bangalore, Mangalore, Delhi, Coimbatore, Hyderabad, Mumbai (Bombay), Ahmedabad, Pune, and Chennai (Madras). A large number of Malayalis have also emigrated to the Middle East, the United States, and Europe. Accessed November 22, 2014.</ref> including a large number of professionals. There were 7,093 Malayalam speakers in Australia in 2006. The 2001 Canadian census reported 7,070 people who listed Malayalam as their mother tongue, mostly in the Greater Toronto Area and Southern Ontario. In 2010, the Census of Population of Singapore reported that there were 26,348 Malayalees in Singapore. The 2006 New Zealand census reported 2,139 speakers. 134 Malayalam speaking households were reported in 1956 in Fiji. There is also a considerable Malayali population in the Persian Gulf regions, especially in Bahrain, Muscat, Doha, Dubai, Abu Dhabi, Kuwait and European region mainly in London.  World Malayalee Council, the organisation working with the Malayali diaspora across the Globe has embarked upon a project for making a data bank of the diaspora. CANNOTANSWER", 
      "qas": [{
        "followup": "m", "yesno": "x", "question": "Where is Malayali located?", 
        "answers": [{
          "text": "30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India,", 
          "answer_start": 51}], 
        "id": "C_69758fcdfc1f46baba0e92c0f3b0919c_1_q#0", 
        "orig_answer": {
          "text": "30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India,",
          "answer_start": 51

...

}
```
