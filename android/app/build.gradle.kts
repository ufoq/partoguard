plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

import java.util.Properties

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

android {
    namespace = "com.partoguard.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.mawges.partoguard"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        vectorDrawables { useSupportLibrary = true }
    }

    signingConfigs {
        create("release") {
            val props = Properties()
            val lp = rootProject.file("local.properties")
            if (lp.exists()) props.load(lp.reader())
            val sf = props.getProperty("signing.storeFile") ?: System.getenv("SIGNING_STORE_FILE")
            if (sf != null) {
                storeFile = file(sf)
                storePassword = props.getProperty("signing.storePassword") ?: System.getenv("SIGNING_STORE_PASSWORD") ?: ""
                keyAlias = props.getProperty("signing.keyAlias") ?: System.getenv("SIGNING_KEY_ALIAS") ?: ""
                keyPassword = props.getProperty("signing.keyPassword") ?: System.getenv("SIGNING_KEY_PASSWORD") ?: ""
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    sourceSets {
        getByName("main") {
            kotlin.srcDirs("src/main/kotlin")
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.navigation.compose)

    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    implementation(libs.coil.compose)
    implementation(libs.okhttp)
    implementation(libs.litertlm.android)
    implementation(project(":llama"))

    debugImplementation(libs.androidx.ui.tooling)
}
