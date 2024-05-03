<template>
    <div style="width: 600px;">
        <div style="font-size: x-large; margin-bottom: 10px; color: #409EFF;">Anomaly Detection</div>
        <el-card style="width: 600px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <el-input v-model="inputStr" type="textarea" style="margin-right: 20px;"></el-input>
                <el-button :icon="Search" circle @click="parseSeqAndDetect" />
            </div>
        </el-card>
        <div style="font-size: large; margin-top: 15px; font-weight: bold;">Session Loss</div>
        <div id="line-chart-container">
            <div style="width: 100%; height: 300px" ref="lineChartRef"></div>
        </div>
        <div style="box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
            <el-table border :data="topk_preds" max-height="310px" height="310px" @cell-click="copyCell">
                <el-table-column label="Window" width="300">
                    <template #default="{ row }">
                        {{ showWindow(row.window) }}
                    </template>
                </el-table-column>
                <el-table-column label="Label" width="100">
                    <template #default="{ row }">
                        E{{ row.label }}
                    </template>
                </el-table-column>
                <el-table-column label="Loss">
                    <template #default="{ row }">
                        {{ row.loss }}
                    </template>
                </el-table-column>
            </el-table>
        </div>
    </div>

</template>

<script setup>
import { Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from "axios";
import * as echarts from 'echarts';
import { onMounted, ref } from "vue";

const copyCell = (row, column, cell, event) => {
    const window = showWindow(row.window)
    navigator.clipboard.writeText(window)
        .then(() => {
            ElMessage({
                message: 'Copied window to clipboard',
                type: 'success',
                plain: true,
            })
        })
}

const showWindow = (window) => {
    let formattedString = "";
    for (let i = 0; i < window.length - 1; i++) {
        formattedString += `E${window[i]}`;
        formattedString += ",";
    }
    formattedString += `E${window[window.length - 1]}`;
    return formattedString;
}

// const inputStr = ref("E5,E5,E22,E5,E11,E9,E11,E9,E11,E9,E26,E26,E26,E2,E4,E4,E4,E4,E4,E4,E4,E4,E3,E4,E4,E4,E4,E4,E4,E4,E2,E2,E23,E23,E23,E21,E21,E21")
const inputStr = ref("E5,E22,E5,E5,E11,E9,E11,E9,E11,E9,E26,E26,E26,E2,E2,E2,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E4,E3,E23,E23,E23,E21,E21,E28,E26,E21")
const parseSeqAndDetect = async () => {
    const separated = inputStr.value.split(",");
    const session = separated.map((eid) => Number(eid.slice(1)))

    const data = { "session": session }
    const resp = await axios.post("http://localhost:5000/detect", data)
    // console.log(resp.data)
    lineChartData.value = resp.data.losses;

    topk_preds.value = []
    for (let i = 0; i < resp.data.topk_preds.length; ++i) {
        topk_preds.value.push({
            "loss": resp.data.losses[i],
            "pred": resp.data.topk_preds[i],
            "prob": resp.data.topk_values[i],
            "label": session[10 + i],
            "window": session.slice(i, i + 10),
        });
    }

    for (let i = 0; i < 10; i++) {
        lineChartData.value.unshift(0);
    }

    option.series[0].data = lineChartData.value;
    option.xAxis.data = session.map((item) => `E${item}`)
    chart.setOption(option);
}

let chart;
let option;
const lineChartRef = ref();
const lineChartData = ref()
const topk_preds = ref([])

onMounted(async () => {
    chart = echarts.init(lineChartRef.value);
    option = {
        xAxis: {
            type: 'category',
            data: [],
            axisTick: {
                show: false,
            },
        },
        yAxis: {
            type: 'value'
        },
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c}',
        },
        series: [
            {
                data: [],
                type: 'line'
            }
        ]
    };
    chart.setOption(option);

    await parseSeqAndDetect();
})

</script>

<style scoped></style>