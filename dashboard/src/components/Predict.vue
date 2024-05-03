<template>
    <div>
        <div style="font-size: x-large; margin-bottom: 10px; color: #409EFF;">Online Prediction</div>
        <!-- <el-button type="danger" plain @click="clear">Clear</el-button> -->
        <el-card style="width: 500px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <el-input v-model="inputValue" type="textarea" style="margin-right: 20px;"></el-input>
                <el-button :icon="Search" circle @click="predict" />
            </div>
        </el-card>
        <div style="box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 500px;">
            <el-table :data="templates" border height="250" style="margin-top: 20px;">
                <el-table-column prop="EventId" label="EventId" width="100" />
                <el-table-column prop="EventTemplate" label="EventTemplate" width="1000" />
            </el-table>
        </div>
        <div style="font-size: large; margin-top: 15px; font-weight: bold;">Loglizer Top K Prediction</div>
        <div id="chart-container">
            <div style="width: 100%; height: 300px" ref="chartRef"></div>
        </div>
        <div style="font-size: large; margin-bottom: 7px; font-weight: bold;">Abstract Trace</div>
        <div style="font-size: large; display: flex; justify-content: center;">
            {{ showTrace(trace) }}
        </div>
    </div>
</template>


<script setup>
import { Search } from '@element-plus/icons-vue'
import axios from "axios";
import { onMounted, ref } from "vue";
import * as echarts from 'echarts';

const templates = ref([])
const trace = ref([0])
const inputValue = ref("E22,E5,E5,E5,E26,E26,E26,E11,E9,E11")

const predict = async () => {
    const separated = inputValue.value.split(",");
    const eids = separated.map((eid) => Number(eid.slice(1)))
    if (eids.length !== 10) {
        ElMessage.error("Please enter a sequence of length 10.")
    } else {
        const data = { data: eids }
        const resp = await axios.post("http://localhost:5000/predict", data)
        pieData = resp.data.topk_pred
        option.series[0].data = pieData
        chart.setOption(option);

        trace.value = resp.data.trace;
        trace.value.unshift(0);
    }
}

const showTrace = (trace) => {
    let formattedString = "";
    for (let i = 0; i < trace.length - 1; i++) {
        formattedString += `S${trace[i]}`;
        formattedString += '→';
    }
    formattedString += `S${trace[trace.length - 1]}`;
    return formattedString;
}

let chart;
let option;
const chartRef = ref(null);
let pieData = [];

onMounted(async () => {
    const resp = await axios.get("http://localhost:5000/static/templates.json");
    templates.value = resp.data

    chart = echarts.init(chartRef.value);
    option = {
        series: [
            {
                type: 'pie',
                radius: ['50%', '70%'],
                label: {
                    show: true,
                    position: 'outside',
                    formatter: '{b} {d}%', // 显示百分比和名称
                },
                data: pieData,
            },
        ],
    };
    chart.setOption(option);

    await predict();
})
</script>

<style scoped></style>