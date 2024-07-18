---
title: "Making Meaningful Visualizations"
layout: single
author_profile: true
author: Christina Kratsch
lecture_name: "Data Science"
lecture_desc: "Grundlegende Methoden für die Exploration und das Management von Daten."
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
classes: wide
---

> "If you torture the data long enough, it will confess to anything" -- [Ronald Coase](https://en.wikiquote.org/wiki/Ronald_Coase)

One of the basic skills of a data scientist is the ability to create a meaningful data visualization that helps answer questions you might have. Prior to visualizing your data, you need to ensure that it has been cleaned and prepared, as you did in prior lessons. After that, you can start deciding how best to present the data.

In this lesson, you will review:

1. How to choose the right chart type
2. How to avoid deceptive charting
3. How to work with color
4. How to style your charts for readability
5. How to build animated or 3D charting solutions
6. How to build a creative visualization

## Choose the right chart type

In previous lessons, you experimented with building all kinds of interesting data visualizations using Matplotlib and Seaborn for charting. In general, you can select the [right kind of chart](https://chartio.com/learn/charts/how-to-select-a-data-vizualization/) for the question you are asking using this table:

| You need to:               | You should use:                 |
| -------------------------- | ------------------------------- |
| Show data trends over time | Line                            |
| Compare categories         | Bar, Pie                        |
| Compare totals             | Pie, Stacked Bar                |
| Show relationships         | Scatter, Line, Facet, Dual Line |
| Show distributions         | Scatter, Histogram, Box         |
| Show proportions           | Pie, Donut, Waffle              |

> ✅ Depending on the makeup of your data, you might need to convert it from text to numeric to get a given chart to support it.

## Avoid deception

Even if a data scientist is careful to choose the right chart for the right data, there are plenty of ways that data can be displayed in a way to prove a point, often at the cost of undermining the data itself. There are many examples of deceptive charts and infographics!

[![How Charts Lie by Alberto Cairo](./images/tornado.png)](https://www.youtube.com/watch?v=oX74Nge8Wkw "How charts lie")

> 🎥 Click the image above for a conference talk about deceptive charts

This chart reverses the X axis to show the opposite of the truth, based on date:

![bad chart 1](images/bad-chart-1.png)

[This chart](https://media.firstcoastnews.com/assets/WTLV/images/170ae16f-4643-438f-b689-50d66ca6a8d8/170ae16f-4643-438f-b689-50d66ca6a8d8_1140x641.jpg) is even more deceptive, as the eye is drawn to the right to conclude that, over time, COVID cases have declined in the various counties. In fact, if you look closely at the dates, you find that they have been rearranged to give that deceptive downward trend.

![bad chart 2](images/bad-chart-2.jpg)

This notorious example uses color AND a flipped Y axis to deceive: instead of concluding that gun deaths spiked after the passage of gun-friendly legislation, in fact the eye is fooled to think that the opposite is true:

![bad chart 3](images/bad-chart-3.jpg)

This strange chart shows how proportion can be manipulated, to hilarious effect:

![bad chart 4](images/bad-chart-4.jpg)

Comparing the incomparable is yet another shady trick. There is a [wonderful web site](https://tylervigen.com/spurious-correlations) all about 'spurious correlations' displaying 'facts' correlating things like the divorce rate in Maine and the consumption of margarine. A Reddit group also collects the [ugly uses](https://www.reddit.com/r/dataisugly/top/?t=all) of data.

It's important to understand how easily the eye can be fooled by deceptive charts. Even if the data scientist's intention is good, the choice of a bad type of chart, such as a pie chart showing too many categories, can be deceptive.

## Color

You saw in the 'Florida gun violence' chart above how color can provide an additional layer of meaning to charts, especially ones not designed using libraries such as Matplotlib and Seaborn which come with various vetted color libraries and palettes. If you are making a chart by hand, do a little study of [color theory](https://colormatters.com/color-and-design/basic-color-theory)

> ✅ Be aware, when designing charts, that accessibility is an important aspect of visualization. Some of your users might be color blind - does your chart display well for users with visual impairments?

Be careful when choosing colors for your chart, as color can convey meaning you might not intend. The 'pink ladies' in the 'height' chart above convey a distinctly 'feminine' ascribed meaning that adds to the bizarreness of the chart itself.

While [color meaning](https://colormatters.com/color-symbolism/the-meanings-of-colors) might be different in different parts of the world, and tend to change in meaning according to their shade. Generally speaking, color meanings include:

| Color  | Meaning             |
| ------ | ------------------- |
| red    | power               |
| blue   | trust, loyalty      |
| yellow | happiness, caution  |
| green  | ecology, luck, envy |
| purple | happiness           |
| orange | vibrance            |

If you are tasked with building a chart with custom colors, ensure that your charts are both accessible and the color you choose coincides with the meaning you are trying to convey.

## Styling your charts for readability

Charts are not meaningful if they are not readable! Take a moment to consider styling the width and height of your chart to scale well with your data. If one variable (such as all 50 states) need to be displayed, show them vertically on the Y axis if possible so as to avoid a horizontally-scrolling chart.

Label your axes, provide a legend if necessary, and offer tooltips for better comprehension of data.

If your data is textual and verbose on the X axis, you can angle the text for better readability. [Matplotlib](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html) offers 3d plotting, if you data supports it. Sophisticated data visualizations can be produced using `mpl_toolkits.mplot3d`.

![3d plots](images/3d.png)


## Review & Self Study

Here are some articles to read about deceptive data visualization:

https://gizmodo.com/how-to-lie-with-data-visualization-1563576606

http://ixd.prattsi.org/2017/12/visual-lies-usability-in-deceptive-data-visualizations/

Take a look at these interest visualizations for historical assets and artifacts:

https://handbook.pubpub.org/

Look through this article on how animation can enhance your visualizations:

https://medium.com/@EvanSinar/use-animation-to-supercharge-data-visualization-cd905a882ad4


## Credits

**⚑ References** <br> This lesson was taken from the course ["Data Science for Beginners"](https://github.com/microsoft/Data-Science-For-Beginners).
