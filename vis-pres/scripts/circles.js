// First undefine 'circles' so we can easily reload this file.
require.undef('circles');

define('circles', ['d3'], function (d3) {

    function draw(container, data, params, width, height) {

        // Get chart width and height
        width = width || 600;
        height = height || 200;

        // Create the SVG element in which we will draw the chart
        var svg = d3.select(container).append("svg")
            .attr('width', width)
            .attr('height', height)
            .append("g");

        // Create a scale from the source data x range to the screen pixel x range
        var x = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([50, width - 50]);

        // Get an object representing all the circles in the chart
        var circles = svg.selectAll('circle').data(data);

        // Add the circles to the chart
        circles.enter()
            .append('circle')
            .attr("cx", function (d, i) {return x(i);})
            .attr("cy", height / 2)
            .attr("r", 20)
            .style("fill", "#1f77b4")
            .style("opacity", 0.7)
            // Code to handle when mouse goes over circle
            .on('mouseover', function() {
                d3.select(this)
                  .interrupt('fade')
                  .style('fill', '#ff850e')
                  .style("opacity", 1)
                  .attr("r", function (d) {return 1.1 * d + 10;});
            })
            // Code to handle when mouse leaves circle
            .on('mouseout', function() {
                d3.select(this)
                    .transition('fade').duration(500)
                    .style("fill", "#1f77b4")
                    .style("opacity", 0.7)
                    .attr("r", function (d) {return d;});
            })
            .transition().duration(2000)
            .attr("r", function (d) {return d;});
    }

    return draw;
});

element.append('Loaded circles.js');