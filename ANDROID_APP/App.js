import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity, ActivityIndicator, Alert, Platform, ImageBackground, Image, Linking, TextInput } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import axios from 'axios';
import * as DocumentPicker from 'expo-document-picker';
import { Audio } from 'expo-av';

import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const API_BASE_URL = 'http://192.168.0.5:8000';
const WS_BASE_URL = `ws://192.168.0.5:8000/ws/convert`;

// --- Reusable Theming & Utilities ---
const THEME = {
  accent: '#D4AF37', // Vintage Gold
  text: '#EAEAEA',
  subText: '#9E9E9E',
  glassBg: 'rgba(20, 20, 25, 0.45)', // very dark translucent
  glassBorder: 'rgba(255, 255, 255, 0.1)',
};

const bgImage = require('./assets/bg.png'); // Sourced via generate_image!

// Custom Glass Component
const GlassCard = ({ children, style }) => (
  <BlurView intensity={30} tint="dark" style={[styles.glassCard, style]}>
    {children}
  </BlurView>
);

// Singer Avatar Generator
const Avatar = ({ url, name, size = 60 }) => {
  const getInitials = (n) => n ? n.charAt(0).toUpperCase() : '?';
  const getAvatarColor = (n) => {
    const defaultColors = ['#1a2a6c', '#b21f1f', '#fdbb2d', '#4b6cb7', '#182848', '#FF4E50'];
    const idx = n ? n.charCodeAt(0) % defaultColors.length : 0;
    return defaultColors[idx];
  };

  if (url) {
    const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;
    return <Image source={{ uri: fullUrl }} style={{ width: size, height: size, borderRadius: size / 2, borderWidth: 1, borderColor: THEME.accent }} />;
  }
  
  return (
    <View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: getAvatarColor(name), justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: THEME.glassBorder }}>
      <Text style={{ color: '#fff', fontWeight: 'bold', fontSize: size * 0.4 }}>{getInitials(name)}</Text>
    </View>
  );
};


// -------------------------------------------------------------
// 1. HOME / GALLERY SCREEN
// -------------------------------------------------------------
const HomeScreen = () => {
  const [gallery, setGallery] = useState([]);
  const [loading, setLoading] = useState(true);
  const [playingAudio, setPlayingAudio] = useState(null); 
  const soundRef = useRef(null);

  useEffect(() => {
    fetchGallery();
    return () => stopSound();
  }, []);

  const fetchGallery = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/gallery`);
      setGallery(res.data.gallery.reverse());
    } catch (e) {
      console.log('Error fetching gallery', e);
    } finally {
      setLoading(false);
    }
  };

  const playSound = async (url) => {
    try {
      if (soundRef.current) await soundRef.current.unloadAsync();
      setPlayingAudio(url);
      const { sound } = await Audio.Sound.createAsync({ uri: url }, { shouldPlay: true });
      soundRef.current = sound;
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.didJustFinish) setPlayingAudio(null);
      });
    } catch (e) { /* ignore */ }
  };

  const stopSound = async () => {
    if (soundRef.current) {
      await soundRef.current.unloadAsync();
      setPlayingAudio(null);
    }
  };

  const renderPlayButton = (url, label, icon) => {
    const fullUrl = `${API_BASE_URL}${url}`;
    const isPlaying = playingAudio === fullUrl;
    
    return (
      <TouchableOpacity 
        style={[styles.audioBtn, isPlaying && { borderColor: THEME.accent, backgroundColor: 'rgba(212, 175, 55, 0.2)' }]}
        onPress={() => isPlaying ? stopSound() : playSound(fullUrl)}
      >
        <Ionicons name={isPlaying ? "stop" : icon} size={16} color={isPlaying ? THEME.accent : THEME.text} />
        <Text style={[styles.audioBtnText, isPlaying && { color: THEME.accent }]}> {label}</Text>
      </TouchableOpacity>
    );
  };

  return (
    <ImageBackground source={bgImage} style={styles.bg}>
      <SafeAreaView style={{ flex: 1 }} edges={['top', 'left', 'right']}>
        <Text style={styles.pageTitle}>Conversions</Text>
        {loading ? <ActivityIndicator size="large" color={THEME.accent} style={{marginTop: 50}} /> : (
          <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 120 }}>
            {gallery.length === 0 && <Text style={{color: THEME.subText, textAlign:'center'}}>No conversions recorded.</Text>}
            {gallery.map((item, idx) => (
              <GlassCard key={idx} style={{ marginBottom: 20 }}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15, justifyContent: 'center' }}>
                   <Avatar name={item.source_base} size={40} />
                   <Ionicons name="arrow-forward" size={20} color={THEME.accent} style={{ marginHorizontal: 15 }} />
                   <Avatar name={item.target_base} size={40} />
                </View>
                <Text style={styles.cardHeading}>{item.source_base} → {item.target_base}</Text>
                
                <View style={styles.audioControls}>
                  {renderPlayButton(item.source_url, "Source", "musical-notes")}
                  {renderPlayButton(item.target_url, "Target", "person")}
                  {item.fake_75_url && renderPlayButton(item.fake_75_url, "Raw Extract", "pulse")}
                  {renderPlayButton(item.converted_url, "Final Master", "disc")}
                </View>
                
                {item.metrics ? (
                  <View style={styles.metricsBox}>
                    <Text style={styles.metricsText}>{item.metrics}</Text>
                  </View>
                ) : null}
              </GlassCard>
            ))}
          </ScrollView>
        )}
      </SafeAreaView>
    </ImageBackground>
  );
};


// -------------------------------------------------------------
// 2. UPLOAD SCREEN
// -------------------------------------------------------------
const UploadScreen = () => {
  const [uploading, setUploading] = useState(false);

  const handleUploadSource = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: ['audio/*', 'application/zip'], multiple: true });
      if (result.canceled) return;
      
      setUploading(true);
      const formData = new FormData();
      result.assets.forEach((a) => formData.append('files', { uri: a.uri, name: a.name, type: a.mimeType || 'audio/wav' }));
      await axios.post(`${API_BASE_URL}/upload/source`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      Alert.alert('Success', 'Sources saved!');
    } catch (e) {
      Alert.alert('Error', e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleUploadTarget = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: ['audio/*'], multiple: true });
      if (result.canceled) return;
      
      Alert.alert(
        "Optional Icon?",
        "Do you want to upload a profile picture for this singer target?",
        [
          { text: "No, Skip", onPress: () => performTargetUpload(result.assets, null) },
          { text: "Yes, Select Image", onPress: async () => {
              const imgResult = await DocumentPicker.getDocumentAsync({ type: 'image/*' });
              if (!imgResult.canceled) {
                performTargetUpload(result.assets, imgResult.assets[0]);
              } else {
                performTargetUpload(result.assets, null); // skipped inside picker
              }
          }}
        ]
      );
    } catch (e) { Alert.alert('Error', e.message); }
  };

  const performTargetUpload = async (audioAssets, imageAsset) => {
      setUploading(true);
      try {
        const formData = new FormData();
        audioAssets.forEach((a) => formData.append('files', { uri: a.uri, name: a.name, type: a.mimeType || 'audio/wav' }));
        if (imageAsset) {
          formData.append('icon', { uri: imageAsset.uri, name: imageAsset.name, type: imageAsset.mimeType || 'image/jpeg' });
        }
        await axios.post(`${API_BASE_URL}/upload/target`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
        Alert.alert('Success', 'Target Voice and icon saved!');
      } catch (e) {
        Alert.alert('Error', e.message);
      } finally {
        setUploading(false);
      }
  };

  return (
    <ImageBackground source={bgImage} style={styles.bg}>
      <SafeAreaView style={{ flex: 1, paddingHorizontal: 20 }} edges={['top', 'left', 'right']}>
         <Text style={[styles.pageTitle, { marginLeft: -5 }]}>Library</Text>
         <ScrollView contentContainerStyle={{ paddingBottom: 120 }} showsVerticalScrollIndicator={false}>
           <GlassCard style={{ alignItems: 'center', marginTop: 20 }}>
            <Ionicons name="cloud-upload-outline" size={40} color={THEME.accent} style={{ marginBottom: 10 }} />
            <Text style={styles.cardHeading}>Add Source Media</Text>
            <Text style={{ color: THEME.subText, textAlign: 'center', marginBottom: 20 }}>Upload .wav or .zip files of input vocals to be converted.</Text>
            <TouchableOpacity style={styles.solidBtn} onPress={handleUploadSource} disabled={uploading}>
              <Text style={styles.solidBtnText}>Select Source Files</Text>
            </TouchableOpacity>
         </GlassCard>

         <GlassCard style={{ alignItems: 'center', marginTop: 20 }}>
            <Ionicons name="mic-outline" size={40} color={THEME.accent} style={{ marginBottom: 10 }} />
            <Text style={styles.cardHeading}>Add Target Singer Identity</Text>
            <Text style={{ color: THEME.subText, textAlign: 'center', marginBottom: 20 }}>Upload a .wav clip of the target singer voice. You can optionally upload their photo too!</Text>
            <TouchableOpacity style={styles.solidBtn} onPress={handleUploadTarget} disabled={uploading}>
              <Text style={styles.solidBtnText}>Select Target Files</Text>
            </TouchableOpacity>
         </GlassCard>

         {uploading && <ActivityIndicator size="large" color={THEME.accent} style={{ marginTop: 30 }} />}
         </ScrollView>
      </SafeAreaView>
    </ImageBackground>
  );
};


// -------------------------------------------------------------
// 3. CONVERT SCREEN
// -------------------------------------------------------------
const ConvertScreen = () => {
  const [config, setConfig] = useState({ sources: [], targets: [] });
  const [selectedSources, setSelectedSources] = useState({}); 
  const [selectedTargets, setSelectedTargets] = useState({}); 
  
  const [running, setRunning] = useState(false);
  const [progressLog, setProgressLog] = useState([]);
  const [activeTaskText, setActiveTaskText] = useState('');
  const [overallProgress, setOverallProgress] = useState(0);
  const [finalResults, setFinalResults] = useState([]); // Array of completed payload objects inline
  const wsRef = useRef(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/config`);
      setConfig(res.data);
    } catch (e) { /* silent */ }
  };

  const toggleTarget = (tgtInfo) => {
    const updated = { ...selectedTargets };
    if (updated[tgtInfo.name]) delete updated[tgtInfo.name];
    else updated[tgtInfo.name] = tgtInfo.defaultGender;
    setSelectedTargets(updated);
  };

  const toggleSource = (src) => {
    const updated = { ...selectedSources };
    if (updated[src]) delete updated[src];
    else updated[src] = 'Male';
    setSelectedSources(updated);
  };

  const startConversion = () => {
    if (Object.keys(selectedSources).length === 0 || Object.keys(selectedTargets).length === 0) {
      Alert.alert("Missing", "Select source and target.");
      return;
    }

    const payload = {
      sources: Object.entries(selectedSources).map(([name, gender]) => ({ name, gender })),
      targets: Object.entries(selectedTargets).map(([name, gender]) => ({ name, gender })),
    };

    setRunning(true);
    setProgressLog([]);
    setFinalResults([]);
    setOverallProgress(0);
    setActiveTaskText('Connecting to GPU backend...');

    const ws = new WebSocket(WS_BASE_URL);
    wsRef.current = ws;

    ws.onopen = () => ws.send(JSON.stringify(payload));
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
         setActiveTaskText(data.message);
      } else if (data.type === 'task_start') {
         setActiveTaskText(data.task);
         if (data.overall_progress) setOverallProgress(data.overall_progress);
         // setProgressLog([]); // Retain log or not, let's append
         setProgressLog(prev => [...prev, `\n--- ${data.task} ---`]);
      } else if (data.type === 'step') {
         setProgressLog((prev) => [...prev, data.message]);
      } else if (data.type === 'result_75' || data.type === 'result_100') {
         setProgressLog((prev) => [...prev, `🔸 ${data.message}: ${data.similarity}%`]);
         if (data.type === 'result_100') {
           // Task finished, attach final url and target base internally
           const tgtBase = data.target_base || '';
           
           setFinalResults(prev => [...prev, { url: data.audio_url, tgtBase: tgtBase }]);
         }
      } else if (data.type === 'complete') {
         setRunning(false);
         setActiveTaskText('All operations complete.');
      } else if (data.type === 'error') {
         setRunning(false);
         Alert.alert("Error", data.message);
      }
    };
    
    ws.onerror = () => { setRunning(false); Alert.alert("Error", "WebSocket connection failed."); };
    ws.onclose = () => { setRunning(false); };
  };

  const downloadCSV = (tgtBase) => {
     // Opens the CSV directly using linking
     const url = `${API_BASE_URL}/download_csv/${tgtBase}`;
     Linking.openURL(url);
  };
  
  // Playback component for inline results
  const AudioPlayerInline = ({ url, label }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const audioObj = useRef(null);

    const toggle = async () => {
       if (isPlaying) {
         if (audioObj.current) await audioObj.current.unloadAsync();
         setIsPlaying(false);
       } else {
         const { sound } = await Audio.Sound.createAsync({ uri: `${API_BASE_URL}${url}` }, { shouldPlay: true });
         audioObj.current = sound;
         setIsPlaying(true);
         sound.setOnPlaybackStatusUpdate((st) => { if (st.didJustFinish) setIsPlaying(false); });
       }
    };

    return (
      <TouchableOpacity onPress={toggle} style={[styles.inlineAudioBtn, isPlaying && { borderColor: THEME.accent }]}>
        <Ionicons name={isPlaying ? "pause-circle" : "play-circle"} size={30} color={isPlaying ? THEME.accent : THEME.text} />
        <Text style={{ color: THEME.text, marginLeft: 10 }}>{label}</Text>
      </TouchableOpacity>
    );
  };

  return (
    <ImageBackground source={bgImage} style={styles.bg}>
      <SafeAreaView style={{ flex: 1 }} edges={['top', 'left', 'right']}>
        <Text style={[styles.pageTitle, { paddingHorizontal: 20 }]}>Studio</Text>

        {!running && finalResults.length === 0 ? (
          <ScrollView contentContainerStyle={{ padding: 20, paddingBottom: 120 }}>
             <Text style={styles.sectionTitle}>1. Target Profiles</Text>
             <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 20 }}>
               {config.targets.map(tgt => (
                 <TouchableOpacity 
                    key={tgt.name}
                    style={[styles.targetAvatarWrapper, selectedTargets[tgt.name] && styles.targetAvatarWrapperActive]}
                    onPress={() => toggleTarget(tgt)}
                 >
                    <Avatar url={tgt.icon} name={tgt.name} size={65} />
                    <Text style={styles.avatarLabel} numberOfLines={1}>{tgt.name.replace('_ref.wav', '')}</Text>
                 </TouchableOpacity>
               ))}
             </ScrollView>

             <Text style={styles.sectionTitle}>2. Source Media</Text>
             {config.sources.map(src => (
                <GlassCard key={src} style={[styles.sourceListCard, selectedSources[src] && { borderColor: THEME.accent }]}>
                  <TouchableOpacity style={{ flex: 1 }} onPress={() => toggleSource(src)}>
                    <Text style={{ color: THEME.text, fontSize: 16 }}>{src}</Text>
                  </TouchableOpacity>
                  {selectedSources[src] && (
                     <View style={styles.genderToggle}>
                       <TouchableOpacity onPress={() => setSelectedSources({...selectedSources, [src]: 'Male'})}>
                         <Text style={[styles.gText, selectedSources[src] === 'Male' && styles.gTextActive]}>M</Text>
                       </TouchableOpacity>
                       <Text style={{color:'#666'}}> | </Text>
                       <TouchableOpacity onPress={() => setSelectedSources({...selectedSources, [src]: 'Female'})}>
                         <Text style={[styles.gText, selectedSources[src] === 'Female' && styles.gTextActive]}>F</Text>
                       </TouchableOpacity>
                     </View>
                  )}
                </GlassCard>
             ))}

             <TouchableOpacity style={[styles.solidBtn, { marginTop: 30 }]} onPress={startConversion}>
                <LinearGradient colors={['#D4AF37', '#B58500']} style={styles.gradientBtn}>
                   <Text style={[styles.solidBtnText, { color: '#000' }]}>INITIALIZE ENGINE</Text>
                </LinearGradient>
             </TouchableOpacity>
          </ScrollView>
        ) : (
          /* CONVERSION PROGRESS VIEW (INLINE RESULTS) */
          <View style={{ flex: 1, padding: 20 }}>
             <Text style={{ color: THEME.accent, fontSize: 18, fontWeight: 'bold' }}>{activeTaskText}</Text>
             <View style={styles.progressTrack}>
                <View style={[styles.progressFill, { width: `${overallProgress * 100}%` }]} />
             </View>
             
             <GlassCard style={{ flex: 1, padding: 10, marginBottom: 20 }}>
               <ScrollView>
                 {progressLog.map((log, idx) => (
                   <Text key={idx} style={{ color: log.includes('---') ? THEME.accent : '#aaa', fontFamily: 'monospace', fontSize: 12, marginVertical: 2 }}>{log}</Text>
                 ))}

                 {/* FINAL RESULTS INLINE RENDERED */}
                 {finalResults.map((res, idx) => (
                    <View key={idx} style={styles.inlineResultBox}>
                      <Text style={{ color: '#fff', marginBottom: 10, fontWeight: 'bold' }}>Task Completed! Listen below:</Text>
                      <AudioPlayerInline url={res.url} label="Play Rendered Master" />
                      
                      <TouchableOpacity style={styles.csvBtn} onPress={() => downloadCSV(res.tgtBase)}>
                        <Ionicons name="download-outline" size={18} color={THEME.accent} />
                        <Text style={styles.csvBtnText}> GET IDENTITY PROFILE (.CSV)</Text>
                      </TouchableOpacity>
                    </View>
                 ))}
               </ScrollView>
             </GlassCard>

             {!running && (
               <TouchableOpacity style={[styles.solidBtn, { backgroundColor: '#333' }]} onPress={() => { setFinalResults([]); setProgressLog([]); fetchConfig(); }}>
                 <Text style={styles.solidBtnText}>Start New Batch</Text>
               </TouchableOpacity>
             )}
          </View>
        )}

      </SafeAreaView>
    </ImageBackground>
  );
};


// -------------------------------------------------------------
// 4. RAGA SUITE SCREEN (Swara, Raga, Theme)
// -------------------------------------------------------------
import * as FileSystem from 'expo-file-system/legacy';
import { LogBox } from 'react-native';

LogBox.ignoreLogs([
  'Expo AV has been deprecated',
  'Method readAsStringAsync imported from "expo-file-system" is deprecated'
]);

const RagaSuiteScreen = () => {
  const [activeTool, setActiveTool] = useState(null); // 'swara', 'raga', 'theme'
  
  // -- Swara State --
  const [swaraUploading, setSwaraUploading] = useState(false);
  const [swaraResult, setSwaraResult] = useState(null);

  // -- Raga Predict State --
  const [ragaInput, setRagaInput] = useState('');
  const [ragaResult, setRagaResult] = useState(null);
  const [ragaPredicting, setRagaPredicting] = useState(false);

  // -- Theme State --
  const [themeConverting, setThemeConverting] = useState(false);
  const [themeProgress, setThemeProgress] = useState([]);
  const [themeResult, setThemeResult] = useState(null);

  const handleSwaraUpload = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({ type: ['audio/*'] });
      if (res.canceled) return;
      setSwaraUploading(true);
      const formData = new FormData();
      formData.append('file', { uri: res.assets[0].uri, name: res.assets[0].name, type: res.assets[0].mimeType || 'audio/wav' });
      
      const out = await axios.post(`${API_BASE_URL}/api/swara/extract`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setSwaraResult(out.data);
    } catch (e) { Alert.alert('Error', e.message); }
    finally { setSwaraUploading(false); }
  };

  const handleRagaPredict = async () => {
    if (!ragaInput) return;
    try {
      setRagaPredicting(true);
      const swaras = ragaInput.split(',').map(s => s.trim().toUpperCase());
      const res = await axios.post(`${API_BASE_URL}/api/raga/predict`, { swaras });
      setRagaResult(res.data);
    } catch (e) { Alert.alert('Error', e.message); }
    finally { setRagaPredicting(false); }
  };

  const handleThemeUpload = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({ type: ['audio/*'] });
      if (res.canceled) return;
      
      Alert.alert(
        "Select Target Emotion",
        "Which emotion do you want to convert this to?",
        [
          { text: "Happy", onPress: () => runThemeWs(res.assets[0], "Happy") },
          { text: "Sad", onPress: () => runThemeWs(res.assets[0], "Sad") },
          { text: "Angry", onPress: () => runThemeWs(res.assets[0], "Angry") },
          { text: "Peaceful", onPress: () => runThemeWs(res.assets[0], "Peaceful") },
          { text: "Cancel", style: "cancel" }
        ]
      );
    } catch (e) { Alert.alert('Error', e.message); }
  };

  const runThemeWs = async (fileAsset, emotion) => {
    setThemeConverting(true);
    setThemeProgress([]);
    setThemeResult(null);
    try {
      const b64 = await FileSystem.readAsStringAsync(fileAsset.uri, { encoding: 'base64' });
      const ws = new WebSocket(`${WS_BASE_URL.replace('/ws/convert', '/ws/theme_convert')}`);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({ filename: fileAsset.name, target_emotion: emotion, audio_bytes_base64: b64 }));
      };
      
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === 'step') {
          setThemeProgress(prev => [...prev, d.message]);
        } else if (d.type === 'complete') {
          setThemeResult(d);
          setThemeConverting(false);
        } else if (d.type === 'error') {
          Alert.alert("Error", d.message);
          setThemeConverting(false);
        }
      };
    } catch (e) {
      Alert.alert("Error", e.message);
      setThemeConverting(false);
    }
  };

  // --- RENDERS ---
  const renderMenu = () => (
    <View style={{ flex: 1, padding: 20 }}>
      <Text style={styles.sectionTitle}>Select a Core AI Module</Text>
      
      <TouchableOpacity onPress={() => setActiveTool('swara')}>
        <GlassCard style={{ marginBottom: 20, alignItems: 'center' }}>
          <Ionicons name="pulse" size={35} color={THEME.accent} />
          <Text style={[styles.cardHeading, { marginTop: 10 }]}>1. Swara Extractor</Text>
          <Text style={{ color: THEME.subText, textAlign: 'center' }}>Extract pitch profile and sequence from any audio.</Text>
        </GlassCard>
      </TouchableOpacity>
      
      <TouchableOpacity onPress={() => setActiveTool('raga')}>
        <GlassCard style={{ marginBottom: 20, alignItems: 'center' }}>
          <Ionicons name="git-merge" size={35} color={THEME.accent} />
          <Text style={[styles.cardHeading, { marginTop: 10 }]}>2. Raga Predictor</Text>
          <Text style={{ color: THEME.subText, textAlign: 'center' }}>FP-Growth association rule mining for Swara patterns.</Text>
        </GlassCard>
      </TouchableOpacity>
      
      <TouchableOpacity onPress={() => setActiveTool('theme')}>
        <GlassCard style={{ marginBottom: 20, alignItems: 'center' }}>
          <Ionicons name="color-wand" size={35} color={THEME.accent} />
          <Text style={[styles.cardHeading, { marginTop: 10 }]}>3. Theme Converter</Text>
          <Text style={{ color: THEME.subText, textAlign: 'center' }}>Convert song emotion using DSP and Phase Vocoder.</Text>
        </GlassCard>
      </TouchableOpacity>
    </View>
  );

  return (
    <ImageBackground source={bgImage} style={styles.bg}>
      <SafeAreaView style={{ flex: 1, paddingHorizontal: 20 }} edges={['top', 'left', 'right']}>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10, marginTop: 10 }}>
          {activeTool && (
            <TouchableOpacity onPress={() => setActiveTool(null)} style={{ marginRight: 15 }}>
              <Ionicons name="arrow-back" size={28} color={THEME.accent} />
            </TouchableOpacity>
          )}
          <Text style={[styles.pageTitle, { marginTop: 0, marginBottom: 0 }]}>Raga Suite</Text>
        </View>

        {!activeTool && renderMenu()}

        {activeTool === 'swara' && (
          <ScrollView contentContainerStyle={{ paddingBottom: 120 }}>
            <GlassCard style={{ alignItems: 'center' }}>
              <Ionicons name="musical-notes-outline" size={40} color={THEME.accent} />
              <Text style={{ color: '#fff', marginVertical: 10 }}>Upload song to extract Swaras</Text>
              <TouchableOpacity style={[styles.solidBtn, { width: '80%' }]} onPress={handleSwaraUpload}>
                <Text style={styles.solidBtnText}>{swaraUploading ? "Extracting..." : "Upload & Analyze"}</Text>
              </TouchableOpacity>
            </GlassCard>
            
            {swaraResult && (
              <GlassCard style={{ marginTop: 20 }}>
                <Text style={styles.cardHeading}>Analysis Results</Text>
                <Text style={{ color: THEME.subText }}>Detected Raga: {swaraResult.swara_profile.all_raga_scores[0].raga}</Text>
                <Text style={{ color: THEME.subText }}>Duration: {swaraResult.note_sequence.duration_analysed}s</Text>
                <Text style={{ color: THEME.subText, marginTop: 10 }}>Note Flow:</Text>
                <Text style={{ color: '#aaa', fontSize: 12 }}>{swaraResult.note_sequence.note_events.slice(0, 30).join(' → ')}...</Text>
              </GlassCard>
            )}
          </ScrollView>
        )}

        {activeTool === 'raga' && (
          <ScrollView contentContainerStyle={{ paddingBottom: 120 }}>
            <GlassCard>
              <Text style={styles.cardHeading}>Enter Swaras</Text>
              <TextInput 
                style={{ backgroundColor: 'rgba(0,0,0,0.5)', color: '#fff', padding: 15, borderRadius: 10, marginTop: 10, borderWidth: 1, borderColor: THEME.glassBorder }}
                placeholder="e.g. R2, M1, P"
                placeholderTextColor="#666"
                value={ragaInput}
                onChangeText={setRagaInput}
              />
              <TouchableOpacity style={[styles.solidBtn, { marginTop: 15 }]} onPress={handleRagaPredict}>
                <Text style={styles.solidBtnText}>{ragaPredicting ? "Predicting..." : "Predict Next & Match Raga"}</Text>
              </TouchableOpacity>
            </GlassCard>
            
            {ragaResult && (
              <GlassCard style={{ marginTop: 20 }}>
                <Text style={styles.cardHeading}>1. Recommended Next Swaras</Text>
                {ragaResult.recommendations.map((r, i) => (
                  <Text key={i} style={{ color: '#fff' }}>• {r.swara} (Conf: {r.confidence.toFixed(2)})</Text>
                ))}
                
                <Text style={[styles.cardHeading, { marginTop: 20 }]}>2. Raga Matches</Text>
                {ragaResult.ragas.map((r, i) => (
                  <Text key={i} style={{ color: THEME.accent }}>• {r.raga} (Score: {r.score.toFixed(2)})</Text>
                ))}
              </GlassCard>
            )}
          </ScrollView>
        )}

        {activeTool === 'theme' && (
          <ScrollView contentContainerStyle={{ paddingBottom: 120 }}>
            <GlassCard style={{ alignItems: 'center' }}>
              <Ionicons name="color-filter-outline" size={40} color={THEME.accent} />
              <Text style={{ color: '#fff', marginVertical: 10, textAlign: 'center' }}>Select a Tamil song to transform its emotional raga footprint.</Text>
              <TouchableOpacity style={[styles.solidBtn, { width: '80%' }]} onPress={handleThemeUpload}>
                <Text style={styles.solidBtnText}>Select Source Song</Text>
              </TouchableOpacity>
            </GlassCard>
            
            {themeProgress.length > 0 && (
              <GlassCard style={{ marginTop: 20 }}>
                {themeProgress.map((p, i) => (
                  <Text key={i} style={{ color: '#aaa', fontSize: 12, marginVertical: 2 }}>{p}</Text>
                ))}
                {themeConverting && <ActivityIndicator color={THEME.accent} style={{ marginTop: 10 }} />}
              </GlassCard>
            )}
            
            {themeResult && (
              <GlassCard style={{ marginTop: 20, borderColor: THEME.accent }}>
                <Text style={styles.cardHeading}>Transformation Complete!</Text>
                <Text style={{ color: '#fff', marginVertical: 10 }}>Source was {themeResult.report.source_raga}. Target pitch shifted by {themeResult.report.pitch_shift} semitones.</Text>
                
                <TouchableOpacity onPress={async () => {
                   const { sound } = await Audio.Sound.createAsync({ uri: `${API_BASE_URL}${themeResult.audio_url}` }, { shouldPlay: true });
                }} style={[styles.solidBtn, { borderColor: THEME.accent, borderWidth: 1 }]}>
                  <Text style={[styles.solidBtnText, { color: THEME.accent }]}>▶ Play Emotion Render</Text>
                </TouchableOpacity>
              </GlassCard>
            )}
          </ScrollView>
        )}

      </SafeAreaView>
    </ImageBackground>
  );
};


// -------------------------------------------------------------
// APP CONFIG & STYLES
// -------------------------------------------------------------
const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer theme={DarkTheme}>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            headerShown: false,
            tabBarIcon: ({ color, size }) => {
              let iconName = 'home';
              if (route.name === 'Gallery') iconName = 'albums';
              else if (route.name === 'Studio') iconName = 'mic';
              else if (route.name === 'Library') iconName = 'folder';
              else if (route.name === 'Raga Suite') iconName = 'musical-notes';
              return <Ionicons name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: THEME.accent,
            tabBarInactiveTintColor: THEME.subText,
            tabBarStyle: {
              position: 'absolute',
              backgroundColor: THEME.glassBg,
              borderTopWidth: 0,
              elevation: 0,
              height: 65,
              paddingBottom: 10,
              paddingTop: 5,
              borderTopLeftRadius: 20,
              borderTopRightRadius: 20,
              left: 10,
              right: 10,
              bottom: 15,
            },
            tabBarBackground: () => (
               <BlurView tint="dark" intensity={70} style={[StyleSheet.absoluteFill, { borderRadius: 20, overflow: 'hidden' }]} />
            )
          })}
        >
          <Tab.Screen name="Studio" component={ConvertScreen} />
          <Tab.Screen name="Raga Suite" component={RagaSuiteScreen} />
          <Tab.Screen name="Gallery" component={HomeScreen} />
          <Tab.Screen name="Library" component={UploadScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1, resizeMode: 'cover' },
  pageTitle: { fontSize: 32, fontWeight: 'bold', color: '#fff', marginTop: 20, marginBottom: 10, letterSpacing: 1 },
  sectionTitle: { fontSize: 20, color: '#EAEAEA', fontWeight: 'bold', marginTop: 10, marginBottom: 15 },
  
  glassCard: {
    backgroundColor: THEME.glassBg,
    borderRadius: 15,
    padding: 20,
    borderWidth: 1,
    borderColor: THEME.glassBorder,
    overflow: 'hidden',
  },
  cardHeading: { fontSize: 18, fontWeight: 'bold', color: '#fff', marginBottom: 5 },
  
  solidBtn: {
    backgroundColor: '#1E1E1E',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
    overflow: 'hidden',
  },
  gradientBtn: {
    paddingVertical: 15,
    alignItems: 'center',
    justifyContent: 'center',
  },
  solidBtnText: { color: '#fff', fontSize: 16, fontWeight: 'bold', textAlign: 'center', margin: 15 },

  // Audio Buttons
  audioControls: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginTop: 10 },
  audioBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#333',
  },
  audioBtnText: { color: THEME.text, fontSize: 12, fontWeight: '600' },
  
  metricsBox: { marginTop: 15, padding: 10, backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 8, borderWidth: 1, borderColor: '#222' },
  metricsText: { color: THEME.accent, fontSize: 10, fontFamily: 'monospace' },

  // Avatar Carousel
  targetAvatarWrapper: { alignItems: 'center', marginRight: 15, opacity: 0.5 },
  targetAvatarWrapperActive: { opacity: 1, transform: [{ scale: 1.1 }] },
  avatarLabel: { color: '#fff', fontSize: 10, marginTop: 5, maxWidth: 65, textAlign: 'center' },

  sourceListCard: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 15, marginBottom: 10 },
  genderToggle: { flexDirection: 'row' },
  gText: { color: THEME.subText, fontSize: 16 },
  gTextActive: { color: THEME.accent, fontWeight: 'bold' },

  // Progress
  progressTrack: { height: 4, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 2, marginVertical: 15 },
  progressFill: { height: '100%', backgroundColor: THEME.accent, borderRadius: 2 },

  // Inline Results
  inlineResultBox: { marginTop: 20, padding: 15, backgroundColor: 'rgba(212, 175, 55, 0.1)', borderRadius: 10, borderWidth: 1, borderColor: THEME.accent },
  inlineAudioBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#000', padding: 10, borderRadius: 10, marginBottom: 10 },
  csvBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 10, borderColor: THEME.accent, borderWidth: 1, borderRadius: 8 },
  csvBtnText: { color: THEME.accent, fontWeight: 'bold', fontSize: 12 }
});
