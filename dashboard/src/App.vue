<script setup>
import Header from './components/Header.vue';
import Predict from './components/Predict.vue';
import Diagram from './components/Diagram.vue';
import Data from './components/Data.vue';
</script>

<template>
    <div class="container" style="background: #fafafa; height: 100%;">
        <el-container style="height: 100%;">
            <el-header style="padding: 0;">
                <Header></Header>
            </el-header>
            <el-main style="padding: 0; display: flex;">
                <Predict></Predict>
                <Diagram :width="960" :height="720"></Diagram>
            </el-main>
            <el-footer height="300">
                <Data></Data>
            </el-footer>
        </el-container>
    </div>
</template>

<style>
html,
body,
#app {
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
}
</style>
