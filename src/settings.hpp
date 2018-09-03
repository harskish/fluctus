#pragma once

#include <string>
#include <map>
#include "json.hpp"

class Settings
{
public:
    // Singleton pattern
    static Settings &getInstance() {
        static Settings instance;
        return instance;
    }
    Settings(Settings const&) = delete;
    void operator=(Settings const&) = delete;

    // Getters
    std::string getPlatformName() { return platformName; }
    std::string getDeviceName() { return deviceName; }
    std::string getEnvMapName() { return envMapName; }
    void setEnvMapName(const std::string name) { envMapName = name; };
    std::map<unsigned int, std::string> getShortcuts() { return shortcuts; }
    int getWindowWidth() { return windowWidth; };
    int getWindowHeight() { return windowHeight; };
    float getRenderScale() { return renderScale; };
    void setRenderScale(float s) { renderScale = s; };
    bool getUseBitstack() { return clUseBitstack; }
    bool getUseSoA() { return clUseSoA; }
    unsigned int getWfBufferSize() { return wfBufferSize; }

private:
    Settings();
    void init();
    void load();
    void import(nlohmann::json j);

    // Contents of settings singleton
    std::string platformName;
    std::string deviceName;
    std::string envMapName;
    std::map<unsigned int, std::string> shortcuts;
    unsigned int wfBufferSize;
    bool clUseBitstack;
    bool clUseSoA;
    int windowWidth;
    int windowHeight;
    float renderScale;
};
