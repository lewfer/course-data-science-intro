// First undefine 'scatter' so we can easily reload this file.
require.undef('scatter');

define('scatter', ['d3'], function (d3) {

    function draw(container, data, params, width, height) {
        var jParams = JSON.parse(params);
        
        // Get chart width and height
        width = width || 600;
        height = height || 200;
        
        // Get chart parameters passed in
        var circleSize = jParams["size"] || 15;
        var padding = jParams["padding"] || 20;
        
        // Create the SVG element in which we will draw the chart
        var svg = d3.select(container).append("svg")
            .attr('width', width)
            .attr('height', height)
            .append("g");

        // Create a scale from the source data x range to the screen pixel x range
        var xScale = d3.scaleLinear()
                        .domain([0, d3.max(data, function(d) {return d[0];})])
                        .range([padding,width-padding]);
        
        // Create a scale from the source data y range to the screen pixel y range  
        var yScale = d3.scaleLinear()
                        .domain([0, d3.max(data, function(d) {return d[1];})])
                        .range([height-padding,padding]);        
        

        // Get an object representing all the circles in the chart
        var circles = svg.selectAll('circle').data(data);

        // Add the circles to the chart
        circles.enter()
            .append('circle')
            .attr("cx", function (d, i) {return xScale(d[0]);})
            .attr("cy", function (d, i) {return yScale(d[1]);})
            .attr("r", circleSize)
            .style("fill", "#1f77b4")
            .style("opacity", 0.7)
            // Code to handle when mouse goes over circle
            .on('mouseover', function() {
                d3.select(this)
                  .interrupt('fade')
                  .style('fill', '#ff850e')
                  .style("opacity", 1);
            })
            // Code to handle when mouse leaves circle
            .on('mouseout', function() {
                d3.select(this)
                    .transition('fade').duration(500)
                    .style("fill", "#1f77b4")
                    .style("opacity", 0.7);
            });
        
        /*
        // Add a label to each circle
        var text = svg.selectAll('text').data(data);
        
        text.enter()
            .append("text")
            .text(function(d) {return d[0] + "," + d[1];})
            .attr("x", function (d, i) {return xScale(d[0]);})
            .attr("y", function (d, i) {return yScale(d[1]);});
            */
        
        // Add x axis
        var xAxis = d3.axisBottom()
            .scale(xScale)
            .ticks(5);
        svg.append("g")
            .attr("transform", "translate(0," + (height - padding) + ")")
            .call(xAxis);

        // Add y axis
        var yAxis = d3.axisLeft()
                .scale(yScale)
                .ticks(5);
        svg.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(" + padding + ",0)")
            .call(yAxis);        
    }

    return draw;
});

element.append('Loaded scatter.js');