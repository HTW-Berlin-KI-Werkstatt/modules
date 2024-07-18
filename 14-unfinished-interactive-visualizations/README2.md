## Animation and 3D chart display

Some of the best data visualizations today are animated. Shirley Wu has amazing ones done with D3, such as '[film flowers](http://bl.ocks.org/sxywu/raw/d612c6c653fb8b4d7ff3d422be164a5d/)', where each flower is a visualization of a movie. Another example for the Guardian is 'bussed out', an interactive experience combining visualizations with Greensock and D3 plus a scrollytelling article format to show how NYC handles its homeless problem by bussing people out of the city.

![busing](images/busing.png)

> "Bussed Out: How America Moves its Homeless" from [the Guardian](https://www.theguardian.com/us-news/ng-interactive/2017/dec/20/bussed-out-america-moves-homeless-people-country-study). Visualizations by Nadieh Bremer & Shirley Wu

While this lesson is insufficient to go into depth to teach these powerful visualization libraries, try your hand at D3 in a Vue.js app using a library to display a visualization of the book "Dangerous Liaisons" as an animated social network.

> "Les Liaisons Dangereuses" is an epistolary novel, or a novel presented as a series of letters. Written in 1782 by Choderlos de Laclos, it tells the story of the vicious, morally-bankrupt social maneuvers of two dueling protagonists of the French aristocracy in the late 18th century, the Vicomte de Valmont and the Marquise de Merteuil. Both meet their demise in the end but not without inflicting a great deal of social damage. The novel unfolds as a series of letters written to various people in their circles, plotting for revenge or simply to make trouble. Create a visualization of these letters to discover the major kingpins of the narrative, visually.

You will complete a web app that will display an animated view of this social network. It uses a library that was built to create a [visual of a network](https://github.com/emiliorizzo/vue-d3-network) using Vue.js and D3. When the app is running, you can pull the nodes around on the screen to shuffle the data around.

![liaisons](images/liaisons.png)

## Project: Build a chart to show a network using D3.js

> This lesson folder includes a `solution` folder where you can find the completed project, for your reference.

1. Follow the instructions in the README.md file in the starter folder's root. Make sure you have NPM and Node.js running on your machine before installing your project's dependencies.

2. Open the `starter/src` folder. You'll discover an `assets` folder where you can find a .json file with all the letters from the novel, numbered, with a 'to' and 'from' annotation.

3. Complete the code in `components/Nodes.vue` to enable the visualization. Look for the method called `createLinks()` and add the following nested loop.

Loop through the .json object to capture the 'to' and 'from' data for the letters and build up the `links` object so that the visualization library can consume it:

```javascript
//loop through letters
      let f = 0;
      let t = 0;
      for (var i = 0; i < letters.length; i++) {
          for (var j = 0; j < characters.length; j++) {
              
            if (characters[j] == letters[i].from) {
              f = j;
            }
            if (characters[j] == letters[i].to) {
              t = j;
            }
        }
        this.links.push({ sid: f, tid: t });
      }
  ```

Run your app from the terminal (npm run serve) and enjoy the visualization!

Finally, take a tour of the internet to discover deceptive visualizations. How does the author fool the user, and is it intentional? Try correcting the visualizations to show how they should look.