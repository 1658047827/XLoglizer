<template>
    <div style="margin-left: 20px; margin-right: 20px; display: flex;">
        <div id="diagram-container">
            <div style="font-size: x-large; color: #409EFF;">State Diagram</div>
            <svg id="graph-svg" :width="width" :height="height" :viewBox="[0, 0, width, height]"
                style="max-width: 100%; height: auto; font: 12px sans-serif;"></svg>
            <div v-if="showNodeInfo" id="node-info">
                <el-card style="width: 300px; margin-top: 50px;">
                    <span style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: x-large; color: #202121;">S{{ state }}</span>
                        <el-button circle :icon="Close" size="small" @click="closeNodeInfo"></el-button>
                    </span>
                    <div style="font-size: medium; border-bottom: 2px solid #3375b9;">associated input log keys</div>
                    <el-scrollbar max-height="120px">
                        <div v-for="(value, key) in state_input[state]" :key="key" class="state-x-item">
                            <span style="font-size: large;">E{{ key }}</span>
                            <el-tag type="primary" effect="plain" round style="margin-right: 12px;">
                                count: {{ value }}
                            </el-tag>
                        </div>
                    </el-scrollbar>
                    <el-empty style="padding: 20px;" v-if="isEmpty(state_input[state])" :image-size="50" />
                    <div style="font-size: medium; border-bottom: 2px solid #3375b9;">associated prediction labels</div>
                    <el-scrollbar max-height="120px">
                        <div v-for="(value, key) in state_label[state]" :key="key" class="state-x-item">
                            <span style="font-size: large;">E{{ key }}</span>
                            <el-tag type="primary" effect="plain" round style="margin-right: 12px;">
                                count: {{ value }}
                            </el-tag>
                        </div>
                    </el-scrollbar>
                    <el-empty style="padding: 20px;" v-if="isEmpty(state_label[state])" :image-size="50" />
                </el-card>
            </div>
        </div>
    </div>
</template>

<script setup>
import { Close } from '@element-plus/icons-vue'
import axios from "axios";
import * as d3 from "d3";
import { onMounted, ref } from "vue";

const state = ref(0)
const state_input = ref([])
const state_label = ref([])
const showNodeInfo = ref(false)

const { width, height } = defineProps({
    width: {
        type: Number,
        default: 1080,
    },
    height: {
        type: Number,
        default: 720,
    }
})

const isEmpty = (obj) => {
    return obj === null || obj === undefined || Object.keys(obj).length === 0;
}

const closeNodeInfo = () => {
    showNodeInfo.value = false;
}

const colorScale = d3.scaleLinear()
    .domain([0, 1])
    .range(["#66b1ff", "#213d5b"]);

function loopArc(d) {
    const r = d.target.size;
    return `M ${d.target.x - r} ${d.target.y}
            A ${r} ${r} 1 1 1 ${d.target.x} ${d.target.y - r}`;
}

function linkArc(d) {
    const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
    const offsetX = ((d.target.x - d.source.x) * d.target.size) / r;
    const offsetY = ((d.target.y - d.source.y) * d.target.size) / r;
    return `M ${d.source.x} ${d.source.y}
            A ${r} ${r} 0 0 1 ${d.target.x - offsetX} ${d.target.y - offsetY}`;
}

function drag(simulation) {

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.05).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

function clicked(event, d) {
    state.value = d.id;
    showNodeInfo.value = true;
}

onMounted(async () => {
    const resp0 = await axios.get("http://localhost:5000/static/force.json");
    const graph = resp0.data;
    const selfLoops = graph.links.filter((el) => el.source === el.target)
    const edgeLinks = graph.links.filter((el) => el.source !== el.target)

    const resp1 = await axios.get("http://localhost:5000/static/state_input.json");
    state_input.value = resp1.data;

    const resp2 = await axios.get("http://localhost:5000/static/state_label.json");
    state_label.value = resp2.data;

    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-500))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("x", d3.forceX())
        .force("y", d3.forceY())

    const svg = d3.select("#graph-svg")

    svg.append("defs").selectAll("marker")
        .data(graph.links)
        .join("marker")
        .attr("id", d => `end-arrow-${d.source.id}-${d.target.id}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 10)
        .attr("markerWidth", 8.5)
        .attr("markerHeight", 8.5)
        .attr("orient", "auto")
        .attr("markerUnits", "userSpaceOnUse")
        .append("path")
        .attr("fill", d => colorScale(d.weight))
        .attr("d", "M0,-5L10,0L0,5");

    const loop = svg.append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(selfLoops)
        .join("path")
        .attr("stroke", d => colorScale(d.weight))
        .attr("stroke-width", d => d.weight * 3)
        .attr("marker-end", d => `url(#end-arrow-${d.source.id}-${d.target.id})`)

    const link = svg.append("g")
        .attr("fill", "none")
        .selectAll("path")
        .data(edgeLinks)
        .join("path")
        .attr("stroke", d => colorScale(d.weight))
        .attr("stroke-width", d => d.weight * 3.5)
        .attr("marker-end", d => `url(#end-arrow-${d.source.id}-${d.target.id})`)

    const node = svg.append("g")
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .selectAll("g")
        .data(graph.nodes)
        .join("g")
        .call(drag(simulation));

    node.append("circle")
        .attr("cursor", "pointer")
        .attr("stroke", "white")
        .attr("stroke-width", 1.5)
        .attr("r", d => d.size)
        .attr("fill", "#ff3399")
        .on("click", clicked);

    node.append("title")
        .text(d => `S${d.id}`);

    node.append("text")
        .attr("x", d => d.size)
        .attr("y", "0.31em")
        .text(d => `S${d.id}`)
        .clone(true).lower()
        .attr("fill", "none")
        .attr("stroke", "white")
        .attr("stroke-width", 3);

    simulation.on("tick", () => {
        node.attr("transform", d => `translate(${d.x},${d.y})`);
        loop.attr("d", loopArc)
        link.attr("d", linkArc);
    });
})

</script>

<style scoped>
#diagram-container {
    position: relative;
}

#node-info {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

.state-x-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>