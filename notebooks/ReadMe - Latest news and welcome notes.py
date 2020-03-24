# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ***
# MAGIC 
# MAGIC ## Welcome to the Data Science Collaboration Space, an initiative of the Bertelsmann Data Exchange Project. 
# MAGIC 
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://cm230.github.io/DataScienceCollaborationSpace4.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC __What is it?__<br>
# MAGIC The Data Science Sandbox consists of an Azure Data Lake Store and Azure Databricks, a state-of-the-art, fast cluster-based (enterprise version of Apache Spark), easy-to-use Data Science platform to foster group-wide Data Science collaboration. Its native Microsoft Azure integration supports Azure Active Directory single-sign-on and straightforward Azure Storage, SQL Data Warehouse, Cosmos DB, Microsoft Power BI, Tableau, etc. connectivity. It's centrally provided and budgeted.
# MAGIC 
# MAGIC __What is it not?__<br>
# MAGIC It's not a Bertelsmann data integration platform.
# MAGIC It's also not a Data Science production system, it's a sandbox featuring locally provided and accessible datasets, sandbox SLAs and monthly budget limits.
# MAGIC 
# MAGIC This document is very much constantly developing, so revisit regularly to find out about the latest news.

# COMMAND ----------

# MAGIC %md
# MAGIC __1. News__<br>
# MAGIC __2. Documentation material__<br>
# MAGIC __3. Shared workspace - Showcases & challenges__<br>
# MAGIC __4. Support__<br>

# COMMAND ----------

# MAGIC %md
# MAGIC __1. News__
# MAGIC 
# MAGIC + 31 March 2020 -  The __Data Science Sandbox Challenge II__ will be launched during the __on-line Advanced Analytics Day__. This time round the challenge will be in the area of Natural Language Processing (NLP). The various details will be communicated during the Advanced Analytics Day and via Jostle.
# MAGIC + 20 Feb. 2020 - __Donuts for Developer session on Microsoft Cognitive Computing and ML in Azure__. See Jostle for the event details.
# MAGIC + Dec. 2019 - The AI Expert Group has moved onto the new __Jostle-based community platform__ for the Bertelsmann Tech & Data Community. It replaces the AI Expert Group's Slack channel and will be the community's single most important communication channel going forward.
# MAGIC + 9 Oct. 2019 - ... and the winner is: Amongst a whole bunch of impressive solutions, __Christophe Krech's Sandbox Challenge I solution__ proved the most convincing to the jury, identifying the attribute "D19_Soziales" as the single most powerful explanatory feature and making a strong case.
# MAGIC + 7-9 Oct. 2019 - The first __Bertelsmann AI HackDays__ building upon the results of the AI Hackathon four weeks earlier gives the best performing Hackathon teams and associated business units the chance to develop the Hackathon solutions into more mature solutions, possibly even akin to MVPs.
# MAGIC + 30 Sept. 2019 - The __Data Science Sandbox Challenge I__ featuring five business unit teams battling for the grand prize draws to an end. The solutions and the awards will be presented during the __"Data Week 2019"__ on 9 October at the Founders Foundation in Bielefeld, Germany.
# MAGIC + 4 Sept. 2019 - The first __Bertelsmann AI Hackathon__ featuring 55 hackers working on __five content intelligence use cases__ provided and mentored by Bertelsmann business units __RTL CBC, RTl Nederland, n-tv, Audio Now and AZ Direct__ took place to huge acclaim at the Microsoft office in Cologne. Have a look at an highlight video [here](https://www.youtube.com/watch?time_continue=5&v=XHKUKExlKO8).
# MAGIC + 4 & 11 July 2019 - Data Science Sandbox intro webinars. The webinar recording is available [here](https://register.gotowebinar.com/recording/407134162693194764) (approx. 50 mins.)
# MAGIC + 18 June 2019 - The Data Science Sandbox is launched and made generally available as part of the Data Exchange Program's AI Expert Group kick-off.
# MAGIC + 15 May 2019 - The Data Science Sandbox is presented at the Data Exchange US meeting in New York.
# MAGIC + 17 April 2019 - The Data Science Collaboration Space is featured prominently in [BeNet](https://benet.bertelsmann.com/benet/fs/en/news/rueckblick/nachrichten/news_detail_484436.jsp), the group-wide Bertelsmann intranet.
# MAGIC + 10 April 2019 - The Data Science Collaboration Space launch date (Q2/2019) and an accompanying __Data Science competition__ running throughout the 2nd & 3rd quarter of 2019 are announced during the Bertelsmann Data Exchange Advanced Analytics workstream meeting hosted by Penguin Random House UK in London.
# MAGIC + 10 April 2019 - The launch of two new, closely linked __AI expert groups__ kicking off in Q2/2019 is announced at the Bertelsmann Data Exchange Advanced Analytics workstream meeting hosted by Penguin Random House UK in London. Bertelsmann colleagues interested in joining either the AI expert group business community or the AI expert group developer community, or, indeed, both communities can __contact the two community leads, Daan Odijk__ and __Carsten Moenning__ via email.
# MAGIC + 29 January 2019 - The annual [Bertelsmann Data Exchange Data Summit](https://benet.bertelsmann.com/benet/fs/en/news/rueckblick/nachrichten/news_detail_481106.jsp) at Unter der Linden 1 in Berlin co-hosted by Bert Habets and Rolf Hellermann with around 120 group-wide participants gives, amongst other things, an outlook on the 2019 Data Exchange objectives, including the launch of __AI expert groups__ for the AI business and developer community and the launch of the __Data Science Sandbox__. Both announcements are closely related to each other, with launch dates in the 2nd quarter of 2019.

# COMMAND ----------

# MAGIC %md
# MAGIC __2. Documentation material__
# MAGIC 
# MAGIC + [Data Science Sandbox intro webinar](https://register.gotowebinar.com/recording/407134162693194764) - Recording of the July 2019 intro webinar by Carsten Mönning (approx. 50 mins.).
# MAGIC + [Local Notebook: Azure Databricks in 5 minutes](https://westeurope.azuredatabricks.net/?o=5728379491119130#notebook/2774551119379642/command/2774551119379643) - This notebook walks you through setting up your first cluster and running your first analytics experiment.<br>
# MAGIC + [Donuts for Developers session on Azure Databricks - Slidedeck](https://jam12.sapjam.com/groups/p4Re61f1m93cZJjmAzhXEg/documents/Z65mgy9Z7mWHU8piWBH0Ex/slide_viewer) - Slidedeck entitled "Azure Databricks - A Technical Overview" as presented during last year's Donuts session on Azure Databricks. You will need peoplenet Jam access to be able to retrieve this presentation.
# MAGIC + [Donuts for Developers session on Azure Databricks - Video](https://jam12.sapjam.com/groups/p4Re61f1m93cZJjmAzhXEg/documents/zaCRvEBu9T3f4eeyE3X3pH/video_viewer) - Video recording of the complete 2018 Donuts session on Azure Databricks. You will need peoplenet Jam access to be able to retrieve this video.
# MAGIC + [YouTube: Azure Databricks: A Brief Introduction](https://www.youtube.com/watch?v=cxyUy1bZ9mk) - 30min video intro to Azure Databricks.<br>
# MAGIC + [Databricks: Video resources](https://databricks.com/resources/type/product-videos) - Collection of product videos.<br>
# MAGIC + [Databricks: On-line documentation](https://docs.azuredatabricks.net/user-guide/getting-started.html) - The official Azure Databricks documentation.<br>
# MAGIC + [Microsoft: Azure Databricks learning modules](https://docs.microsoft.com/en-us/learn/browse/?products=azure-databricks) - Additional Azure Databricks tutorials dealing with well-defined problem sets, including, for example, Microsoft Power BI integration with Azure Databricks.<br>
# MAGIC + [Microsoft: Azure Databricks learning path](https://docs.microsoft.com/en-us/learn/browse/?products=azure-databricks&resource_type=learning%20path) - Various learning modules neatly organised into learning paths.<br>

# COMMAND ----------

# MAGIC %md
# MAGIC __3. Shared workspace - Showcases & challenges__
# MAGIC 
# MAGIC Navigate to `Workspace\Shared\` to find the following Notebook challenges and showcases in the `\Challenges` and `\Showcases` subfolders, respectively. All you need to know to run any of the showcase Notebooks, for example, in terms of the cluster run-time environment and Python library requirements, is described in the Notebooks. We have given an indication as to what type of Data Science role we believe the showcase or challenge to be of interest to in the list below, but they are indications only and, of course, should not deter anyone from having a closer look at the Notebook under consideration.
# MAGIC 
# MAGIC <img src="https://cm230.github.io/UseCaseTable.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC __4. Support__
# MAGIC 
# MAGIC Please drop us an email at <DataScienceSandbox@bertelsmann.de> in case of any: 
# MAGIC + user access issues
# MAGIC + Sandbox functionality issues, including Data Science library upload requests
# MAGIC + requests for dataset uploads to the Azure Data Lake Store (subject to data privacy & data protection considerations)
# MAGIC + questions related to the showcases or challenges
# MAGIC + ideas for improvement of the environment
# MAGIC 
# MAGIC Please do not use this email address when dealing with individual development or Azure Databricks learning challenges. It's a collaboration environment, so ideally collaborate with others to solve the implementation challenge you might be facing.