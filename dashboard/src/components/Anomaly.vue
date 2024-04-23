<template>
    <div style="width: 600px;">
        <div style="font-size: x-large; margin-bottom: 10px; color: #409EFF;">Anomaly Detection</div>
        <!-- <div style="font-size: medium; margin-top: 10px;">Session Loss</div> -->
        <el-card style="width: 600px; margin-top: 10px;">
            <el-input v-model="inputStr" placeholder="Please input log sequence">
                <template #append>
                    <el-button @click="parseSeqAndDetect" :icon="Search" />
                </template>
            </el-input>
        </el-card>
        <div id="line-chart-container">
            <div style="width: 100%; height: 300px" ref="lineChartRef"></div>
        </div>
        <div style="box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
            <el-table border :data="topk_preds" max-height="360px" height="360px">
                <el-table-column label="Topk Prediction" width="390px">
                    <template #default="{ row }">
                        {{ showPrediction(row.pred) }}
                    </template>
                </el-table-column>
                <el-table-column label="Label">
                    <template #default="{ row }">
                        E{{ row.label }}
                    </template>
                </el-table-column>
                <el-table-column label="Loss" width="250px">
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
import axios from "axios";
import * as echarts from 'echarts';
import { onMounted, ref } from "vue";

const showPrediction = (preds) => {
    let formattedString = "";
    for (let i = 0; i < preds.length - 1; ++i) {
        formattedString += `E${preds[i]}`;
        formattedString += ' > ';
    }
    formattedString += `E${preds[preds.length - 1]}`;
    return formattedString;
}

const inputStr = ref("")
const parseSeqAndDetect = async () => {
    // console.log(inputStr.value)
    const separated = inputStr.value.split(",");
    session.value = separated.map((eid) => Number(eid.slice(1)))
    console.log(session.value)
    await detect();
}

// const session = ref([5, 22, 5, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 23, 23, 23, 21, 21, 28, 26, 21])
const session = ref([5, 5, 22, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26, 2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 23, 23, 23, 21, 21, 21])

let chart;
let option;
const lineChartRef = ref();
const lineChartData = ref()
const topk_preds = ref([])
// const topk_values = ref([])

const detect = async () => {
    const data = { "session": session.value }
    const resp = await axios.post("http://localhost:5000/detect", data)
    console.log(resp.data)
    lineChartData.value = resp.data.losses;

    topk_preds.value = []
    for (let i = 0; i < resp.data.topk_preds.length; ++i) {
        topk_preds.value.push({
            "loss": resp.data.losses[i],
            "pred": resp.data.topk_preds[i],
            "prob": resp.data.topk_values[i],
            "label": session.value[10 + i],
        });
    }

    // topk_preds.value = resp.data.topk_preds;
    // topk_values.value = resp.data.topk_values;

    for (let i = 0; i < 10; i++) {
        lineChartData.value.unshift(0);
    }

    option.series[0].data = lineChartData.value;
    option.xAxis.data = session.value.map((item) => `E${item}`)
    chart.setOption(option);
}

onMounted(async () => {
    chart = echarts.init(lineChartRef.value);
    option = {
        title: {
            text: "Session Loss"
        },
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

    await detect();
})

</script>

<style scoped>
#line-chart-container {
    margin-top: 20px;
}
</style>