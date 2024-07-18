Automated reporting and dashboarding
When dealing with data that is updated periodically and frequently, it's helpful to use automation tools to generate data reports. This saves us the trouble of constantly re-running a repetitive analysis. There are some different ways to deal with sharing data: reports and dashboarding.

Automated reporting options
Reports will usually consist of something like a PDF or an other document (such as MS Word) or a spreadsheet like MS Excel. We already saw how we can do some work with Excel using pandas, but for more powerful control of Excel, we can use other Python packages as well:

xlsxwriter (easily generates charts; this works well with pandas' ExcelWriter)
openpyxl (also allows for charts)
There are also other Excel Python packages. The site http://www.python-excel.org/ is one place with a list of many of these packages. The win32com package can also be used to read and write Excel files on Windows, but its use is not recommended since the documentation for it is lacking. To use Google Sheets through Python, the Google Docs API can be used.

For automated reporting, we have several options. Some solutions use Python as well as non-free licensed software, including ReportLab and Anvil (ReportLab has a freemium model at the time of writing). There are also a few other open source solutions:

FPDF (writes PDF directly)
pdfkit, weasyprint, and xhtml2pdf (these convert HTML to PDF)
PyQt
pylatex (uses LaTeX to create PDFs)
rst2pdf
Similar to Google Sheets, we can use the Google Docs API for Google Docs. For any of these solutions, there is documentation and examples, but the learning curve can be steep for many of them.

One drawback of the PDF or Word document reports is that they are not interactive (but can be printed out if needed). Excel reports can be interactive, but it may not be the best solution for viewing results across different devices such as tablets, phones, and computers. To be able to create multi-platform interactive reports, we can instead use dashboarding.

Automated dashboarding
Dashboarding is used widely in businesses and organizations to monitor results and the status of operations. There are several non-free licensed software options, such as Tableau, Power BI, and many others. These dashboards allow us to easily create interactive plots and combine them into one page or several pages. Then we can share the dashboards easily with others online or by sending them a file. We can also create similar dashboards with open source Python solutions. The advantage of using a licensed solution is that there are often support channels we can call on for help, and they try to make everything as easy as possible to do. The downsides are that it can be expensive and less flexible off the shelf (we can usually combine Python, R, or other languages with dashboarding software such as Tableau, though). With an open source Python dashboarding solution, we can create something for free and it has a lot of flexibility. There are a few different packages for creating dashboards in Python:

Dash
Streamlit
Panel
Voila
There are other related packages, and the list of dashboard packages on PyViz (https://pyviz.org/dashboarding/) will likely be updated as time goes on. In our case, we will use streamlit to quickly create a simple dashboard with data from our payment default dataset.

One note about streamlit: it's not the best solution for streaming data at the time of writing – Dash and Voila are better options for streaming data. This is due to the implementation of streamlit and its available code base, which makes certain streaming data operations difficult or impossible.

We need to first install streamlit, which, at the time of writing, works better with pip than conda due to managing package dependencies (pip install streamlit). Then, we can confirm that it's working by opening a terminal and running streamlit hello, which runs a test application. In our dashboard, we are going to include the main points of our data story.

First, we create a .py file we can use, which we will name default_dashboard.py. We can run the app from a terminal or command line with streamlit run default_dashboard.py (as long as we are in the correct directory with the file). This will show a web address we can click or copy and paste into our browser to view. In our default_dashboard.py file, we start by importing all the necessary packages (and functions), which totals nine:

import phik
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import pycaret.classification as pyclf
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
Next, we set the page configuration to wide:

st.set_page_config(layout="wide")
This can also be done from the webpage where we view the app (from the menu button on the upper right). It has the effect of making the dashboard (with multiple plots) easier to view. Then we load our data and fit a LightGBM model to it:

df = pd.read_excel('data/default of credit card clients.xls',
                    skiprows=1,
                    index_col='ID').sample(1000)
setup = pyclf.setup(df, target='default payment next month', silent=True)
lgbm = pyclf.create_model('lightgbm')
lgbm, tuner = pyclf.tune_model(lgbm, return_tuner=True)
This is similar to what we've seen before, although we are using the silent=True argument so that pycaret doesn't ask for confirmation of the data types. We also get the tuner object back from our tune_model function using the return_tuner argument, which we will use to report our cross-validation accuracy score. We are using a lightgbm model because it was one of the top models with a very similar accuracy to the best model, and it has an easily accessible feature_importances_ attribute we will use soon.

Next, we get the average validation/test set score from our CV and set it as the title of the page:

cv_acc = round(tuner.cv_results_['mean_test_score'].mean(), 3)
st.title(f"CV Accuracy is {cv_acc}")
This sets the HTML title so that our CV accuracy is displayed as large text at the top of the page.

Now we can calculate the PhiK correlation and create a heatmap of the results, create the same bar chart of counts of the PAY_0 column grouped by default status as we did before, and add these to our app:

phik_corr = df.phik_matrix()
correlogram = sns.heatmap(phik_corr)
barchart = px.histogram(df,
                        x='PAY_0',
                        color='default payment next month',
                        barmode='group')
col1, col2 = st.columns(2)
col1.write(correlogram.figure)
col2.write(barchart)
plt.clf()
The first few lines are things we've seen before. The px.histogram function uses Plotly express to create a histogram of the PAY_0 column, and using the default column as the color creates two sets of bars. Using barmode='group' sets the bars next to each other like we had in our earlier plot. Next, we create two columns with streamlit's columns function. Then we use the general-purpose write function from streamlit to add our figures to the columns. For the correlogram or heatmap, we access the figure attribute so that the figure shows up (since it is a matplotlib.axes object). We can directly drop the Plotly chart in the write function, however. Lastly, we use plt.clf() to clear the plot area in preparation for the next plots.

The streamlit, team uses beta_ as a prefix for new functions and features, and eventually the beta_ is removed as the feature becomes mature. For example, the st.set_page_config used to be st.beta_set_page_config.

Next, we can calculate feature importances:

feature_importances = pd.Series(lgbm.feature_importances_,
                                index=pyclf.get_config('X').columns)feat_imp_plot = feature_importances.nlargest(20).plot(kind='barh')
feat_imp_plot.invert_yaxis()
Here, we create a pandas series with our feature importances, and set the index as the column names from our pre-processed data from pycaret. The get_config function allows us to get objects from the setup of our pycaret session, and X is the pre-processed data. Last, we invert the y-axis so that it sorts from greatest to least.

Next, we will create another row for plots (with 2 columns) and add this feature importance plot to the first column:

col1, col2 = st.columns(2)
col1.write(feat_imp_plot.figure)
Now we can create our confusion matrix and add it to the dashboard:

predictions = pyclf.predict_model(lgbm, df)
cm = plot_confusion_matrix(
    confusion_matrix(df['default payment next month'],
    predictions['Label'])
    )
col2.write(cm[1].figure)
We get predictions from our model using the pycaret function which pre-processes the data as we need, for example, one-hot encoding categorical variables. Then we create a confusion matrix plot using mlxtend, giving it the result of the sklearn confusion_matrix function with our true values and predictions. The Label column holds the predicted values from our predictions DataFrame. Finally, we add the chart to our dashboard with col2.write.

Now we can run our dashboard with streamlit run default_dashboard.py. Our resulting dashboard looks like this:


Figure 19.3: The streamlit dashboard of the credit default dataset

We can see we might want to do a little adjusting of the figure shapes and sizes to make it prettier, but it works as a first pass. The plotly bar chart on the upper right is the only interactive part of the page, although we can use plotly or other interactive packages to make more interactive dashboards.

Streamlit was originally made for ML engineers to be able to monitor and investigate their ML processes, although it is being increasingly used for data dashboards as well. At the time of writing, there is an active development team constantly improving streamlit.

With streamlit, we can also add widgets to a sidebar easily with lines of code like default_choices = st.sidebar.multiselect('Choose default categories:', [0, 1], [0, 1]). This will create a multi-select box with options of 0 and 1 (the second argument) and a default of 0 and 1 (the third argument). When we change the selection, the entire .py file reruns from top to bottom, regenerating the page. There are other widgets we can add such as sliders and more.

When something changes in our selections or sidebar in a streamlit app, it reruns the entire file from top to bottom. However, it can mean we reload our data and rerun long ML training processes. There is a solution that streamlit has for this, which is caching, and is described in more detail in the documentation here: https://docs.streamlit.io/en/stable/caching.html.

Streamlit reruns the file if we change settings or change the source .py file. There is also a Rerun button on the app's page, or we can refresh the page to rerun it. However, for other reports or dashboards, we may want to rerun the file at periodic intervals, which we can do in a few different ways.

Scheduling tasks to run automatically
Once we have an automated report or dashboard set up, it can be useful to periodically run the report to get the latest results. This works best for batch applications where we get data updates less frequently. For faster updates, such as streaming data in real time, we would need to use another solution such as the add_rows function of streamlit or another dashboard package such as Dash. There are a few ways to do this:

Cronjobs (in Mac or Linux)
Windows Task Scheduler
Custom Python or other code
Python packages such as schedule
Cronjobs allow us to use the cron software utility (which uses a crontab file) to periodically run some Linux or Mac command. Windows Task Scheduler works similarly, allowing us to run an executable file or terminal command, but has a GUI instead of the command line interface like cron.

We could also write a custom Python function (or something in almost any other programming language) that runs a loop, waits for a specified amount of time (for example, using time.sleep() with Python), and then runs another Python function or file. However, people have made packages to do this already, such as the schedule Python package, which is easy to use based on examples on the readme of the GitHub repository (https://github.com/dbader/schedule) and in the documentation.

If we use a Python file to periodically run a function or another file, we may also want to add this scheduler file to our startup programs on our computer, which can be done by adding files or shortcuts to the start up folder in Windows, through system preferences in Mac, or configuration files in Linux.

There is a lot more automation we can do with Python, and there are several books on the subject. One is Automate the Boring Stuff, by Al Sweigart, along with his other titles on the subject. However, things we've learned in this book can also be leveraged for automation, such as dealing with Excel files and documents, as we learned in Chapter 6, Data Wrangling Documents and Spreadsheets.


Referenz:
https://learning.oreilly.com/library/view/practical-data-science/9781801071970/Text/Chapter_3.xhtml#_idParaDest-87
