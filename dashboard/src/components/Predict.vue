<template>
    <div>
        <el-button type="danger" plain @click="clear" style="margin-top: 50px;">Clear</el-button>
        <el-card shadow="hover" style="width: 500px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <el-tag v-for="eid in eids" :key="eid" closable :disable-transitions="false"
                        @close="handleClose(eid)" style="margin-right: 10px; margin-bottom: 10px;">
                        E{{ eid }}
                    </el-tag>
                    <el-input v-if="inputVisible" ref="InputRef" v-model="inputValue" size="small"
                        @keyup.enter="handleInputConfirm" @blur="handleInputConfirm" style="margin-top: 10px;">
                        <template #prefix>E</template>
                    </el-input>
                    <el-button v-else size="small" @click="showInput" style="margin-bottom: 10px;">
                        + New Event
                    </el-button>
                </div>
                <el-button :icon="Search" circle @click="predict" />
            </div>
        </el-card>
        <el-table :data="templates" border height="250" style="width: 100%; margin-top: 20px;">
            <el-table-column prop="EventId" label="EventId" width="100" />
            <el-table-column prop="EventTemplate" label="EventTemplate" width="400" />
        </el-table>
        <div style="font-size: medium; margin-top: 10px;">RNN Loglizer Top K Prediction</div>
        <div id="chart-container">
            <div style="width: 100%; height: 400px" ref="chartRef"></div>
        </div>
        <div style="font-size: medium; margin-top: 10px;">RNN Loglizer Top K Prediction</div>
    </div>
</template>


<script setup>
import { Search } from '@element-plus/icons-vue'
import axios from "axios";
import { nextTick, onMounted, ref } from "vue";
import * as echarts from 'echarts';

const templates = ref([])

const inputValue = ref("")
const eids = ref([22, 5, 5, 5, 26, 26, 26, 11, 9, 11])
const inputVisible = ref(false)
const InputRef = ref()

const handleClose = (eid) => {
    eids.value.splice(eids.value.indexOf(eid), 1)
}

const showInput = () => {
    inputVisible.value = true
    nextTick(() => {
        InputRef.value.input.focus()
    })
}

const handleInputConfirm = () => {
    if (inputValue.value) {
        eids.value.push(Number(inputValue.value))
    }
    inputVisible.value = false
    inputValue.value = ""
}

const clear = () => {
    eids.value = []
}

const predict = async () => {
    const data = { data: eids.value }
    const resp = await axios.post("http://localhost:5000/predict", data)
    console.log(resp.data.topk_pred)
    pieData = resp.data.topk_pred
    option.series[0].data = pieData
    chart.setOption(option);
}

let chart;
let option;
const chartRef = ref(null);
let pieData = [];

onMounted(async () => {
    const resp = await axios.get("/src/assets/templates.json");
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
})
</script>

<style scoped></style>