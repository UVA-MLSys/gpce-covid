document.addEventListener('DOMContentLoaded', function() {
  //updateLineGraph("SaltLake");
  drawmap();
})



function updateLineGraph(targetFIPS) {
  d3.select('#d3-interactive').html('');
  d3.csv("static/all_counties_manipulated.csv").then(function(data){
    const filteredData = data.filter(row => row.FIPS === targetFIPS);
    drawLineGraph(filteredData);
    console.log(filteredData[0].CountyState);
    d3.select('#county-name').text(filteredData[0].CountyState);
    document.getElementById("CountyName-input").value=filteredData[0].CountyState;
  })}

function getCountyState(targetFIPS,event){
  d3.csv("static/all_counties_manipulated.csv").then(function(data){
    const filteredData = data.filter(row => row.FIPS === targetFIPS);
    showTooltip(filteredData[0].CountyState, event.pageX, event.pageY);
})}


//   // Remove active class from all buttons
//   document.querySelectorAll('.tab-group button').forEach(btn => btn.classList.remove('active'));

//   // Add active class to the clicked button
 
//   const button = document.getElementById(buttonId);
//   button.classList.add('active');
 
//   // Update line graph data based on countyType
//   const data = getDataByCounty(buttonId);

//   // Clear existing line graph
//   d3.select('#d3-interactive').html('');

//   // Draw line graph with updated data
//   drawLineGraph(data);
// }


// function getDataByCounty(buttonId) {
//   // Return the appropriate data based on countyType
//   // You can customize this function to fetch data from an API or use predefined data
//   if (buttonId === 'SaltLake') {
//     d3.csv("static/Utah, Salt Lake.csv").then(data=>{
//       drawLineGraph(data)})
//   } else if (buttonId === 'Hillsborough') {
//     d3.csv("static/Florida, Hillsborough.csv").then(data=>{
//       drawLineGraph(data)})
//   }else if (buttonId === 'Multnomah') {
//     d3.csv("static/Oregon, Multnomah.csv").then(data=>{
//       drawLineGraph(data)})
//   }else if (buttonId === 'Brunswick') {
//     d3.csv("static/Virginia, Brunswick.csv").then(data=>{
//       drawLineGraph(data)})
//   }else if (buttonId === 'Yancey') {
//     d3.csv("static/North Carolina, Yancey.csv").then(data=>{
//       drawLineGraph(data)})
//   }else if (buttonId === 'Ogemaw') {
//     d3.csv("static/Michigan, Ogemaw.csv").then(data=>{
//       drawLineGraph(data)})
//   }
// }

function drawLineGraph(data) {
  console.log(data)
  // Set up the dimensions and margins of the plot
  const width = 800;
  const height = 400;
  const margin = { top: 0, right: 0, bottom: 0, left: 0 };

  // Create the SVG element
  const svg = d3.select('#d3-interactive')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  // Create scales for the x and y axes
  const parseDate = d3.timeParse('%Y-%m-%d');
    data.forEach(d => d.Date = parseDate(d.Date));

    // Create scales for the x and y axes
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.Date))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.max(d.Predicted_Cases, d.Cases))])
      .range([height, 0]);

    // Create x-axis
    const xAxis = d3.axisBottom(xScale);
    svg.append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(xAxis);

    // Create y-axis
    const yAxis = d3.axisLeft(yScale);
    svg.append('g')
      .call(yAxis);


    const legendItems = [
      { label: 'Predicted Cases', color: 'red' },
      { label: 'Actual Cases', color: 'blue' },
      // You can add more legend items for additional lines if needed
  ];

  const legendGroup = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 100}, 0)`);

  const legendEntries = legendGroup.selectAll('.legend-entry')
      .data(legendItems)
      .enter()
      .append('g')
      .attr('class', 'legend-entry')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`);

  legendEntries.append('rect')
      .attr('width', 10)
      .attr('height', 10)
      .attr('fill', d => d.color);

  legendEntries.append('text')
      .attr('x', 15)
      .attr('y', 15)
      .text(d => d.label);

// Create a line generator
const predictedLine = d3.line()
    .x(d => xScale(d.Date))
    .y(d => yScale(d.Predicted_Cases));

  // Create the line path for the predicted cases
  svg.append('path')
    .datum(data)
    .attr('class', 'line predicted')
    .attr('d', predictedLine)
    .attr('fill', 'none') // Remove any fill
    .attr('stroke', 'red') // Customize the color for the actual cases;

  // Create a line generator for the actual cases
  const actualLine = d3.line()
    .x(d => xScale(d.Date))
    .y(d => yScale(d.Cases));

    // Create the line path for the actual cases
    svg.append('path')
      .datum(data)
      .attr('class', 'line actual')
      .attr('d', actualLine)
      .attr('fill', 'none') // Remove any fill
      .attr('stroke', 'blue') // Customize the color for the actual cases;
}


function ready(data){
  var margin={top:0,left:0,right:0,bottom:0},
  height=900-margin.top-margin.bottom,
  width=1200-margin.left-margin.right;

  var projection = d3.geoAlbersUsa()
    .translate([width/2,height/2])
    .scale(1400)
  var path = d3.geoPath().projection(projection)
  var states = topojson.feature(data,data.objects.states).features
  console.log(states);
  var svg  = d3.select("#d3-map")
            .attr("height",height+margin.top+margin.bottom)
            .attr("width",width+margin.left+margin.right)
            .append("g")
            .attr("transform","translate(" + margin.left + "," + margin.top + ")");
  // svg.selectAll(".state")
  // .data(states)
  // .enter().append("path")
  // .attr("class","state")
  // .attr("d",path)


  var counties = topojson.feature(data,data.objects.counties).features
  svg.selectAll(".county")
  .data(counties)
  .enter().append("path")
  .attr("class","county")
  .attr("d",path)
  .attr("FIPS", function(d) {
    return d.id; // Assuming FIPS is a property in the data object
  })
  .on("click", function () {
    // Your onclick event handler code goes here
    document.querySelectorAll('.county').forEach(county => county.classList.remove('active'));
    updateLineGraph(this.getAttribute("FIPS"));
    this.classList.add("active");
  })
  .on("mouseover", function (event) {
    getCountyState(this.getAttribute("FIPS"),event);
    // Show the tooltip box when hovering over a county
  })
  .on("mouseout", function () {
    // Hide the tooltip box when moving the mouse out of a county
    hideTooltip();
  });
}

function showTooltip(countyState, x, y) {
  const tooltip = d3.select("#county-tooltip");

  tooltip.style("left", x + "px")
    .style("top", y + "px")
    .style("display", "block")
    .text(countyState);
}

function hideTooltip() {
  const tooltip = d3.select("#county-tooltip");

  tooltip.style("display", "none");
}

function drawmap (){
  Promise.all([
    d3.json("static/us.json")
    // Add more data loading here if needed
  ]).then(function(data){
    console.log(data);
  ready(data[0])
})}



const inputElement = document.getElementById('CountyName-input');
// Function to be triggered when "Enter" key is pressed
function handleEnterKey(event) {
  if (event.key === 'Enter') {
    searchByCountyName();
  }
}
// Add the event listener to the input element
inputElement.addEventListener('keyup', handleEnterKey);





function toggleHeatmap() {
  const heatmapButton = document.getElementById('heatmap');

  if (heatmapButton.classList.contains('active')) {
      // If the button is active, deactivate it and hide the heatmap
      heatmapButton.classList.remove('active');
      hideHeatmap();
  } else {
      // If the button is not active, activate it and show the heatmap
      heatmapButton.classList.add('active');
      showheatmap();
  }
}

function hideHeatmap() {
  // Get all the county elements on your map
  const countyElements = document.querySelectorAll('.county');
  console.log("a");
  // Reset the fill colors of the county elements
  countyElements.forEach(countyElement => {
      countyElement.style.fill = ""; 
      countyElement.classList.remove("active"); 
  });
}



function showheatmap(){
  d3.csv("static/all_counties_manipulated.csv").then(function(data) {
    // Use d3.group to group the data by FIPS code and calculate total cases
    const groupedData = d3.group(data, d => d.FIPS);
    
    // Create an object to store the total cases for each county
    const totalCasesByFIPS = {};

    // Loop through the grouped data to calculate total cases for each county
    groupedData.forEach((counties, FIPS) => {
      let totalCases = 0;
      counties.forEach(county => {
        totalCases += +county.Cases;
      });
      totalCasesByFIPS[FIPS] = totalCases;
      console.log(totalCasesByFIPS);
    })
    // Get all the county elements on your map
const countyElements = document.querySelectorAll('.county');
countyElements.forEach(countyElement => {countyElement.classList.remove("active")})

// Create a color scale for the heatmap
const colorScale = d3.scaleSequential()
  .domain([0, 500]) // Adjust the domain based on your data
  .interpolator(d3.interpolateReds); // You can use any other color scale you prefer

// Update the fill colors of the county elements based on the totalCases property
countyElements.forEach(countyElement => {
  const FIPS = countyElement.getAttribute("FIPS");
  const totalCases = totalCasesByFIPS[FIPS];

  if (totalCases !== undefined && totalCases !== 0) {
    countyElement.style.fill = colorScale(totalCases);
  } else {
    // If totalCases not found or missing, set a default color
    countyElement.style.fill = "gray";
  }
});
  })}





function searchByCountyName() {
  d3.csv("static/all_counties_manipulated.csv").then(function(data){
    const countyNameInput = document.getElementById("CountyName-input").value;
    const filteredCounty = data.find(row => row.CountyState === countyNameInput);
    const targetFIPS = filteredCounty.FIPS;
    const countyElement = d3.select(`path[FIPS="${targetFIPS}"]`);
    document.querySelectorAll('.county').forEach(county => county.classList.remove('active'));
    updateLineGraph(countyElement.node().getAttribute("FIPS"))
    countyElement.node().classList.add("active");
  })}
