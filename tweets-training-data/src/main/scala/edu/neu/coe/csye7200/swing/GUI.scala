package edu.neu.coe.csye7200.swing

import javax.swing.ImageIcon
import scala.swing._
import scala.swing.event._

object GUI extends SimpleSwingApplication {
  var msg = "original message"
  def top = new MainFrame{
    title = "tweets-traning-data"
    import BorderPanel.Position._
    var textField = new TextField()
    val buttonGroup = new ButtonGroup {
      buttons += new RadioButton("Random Forest Classifier")
      buttons += new RadioButton("Naive Bayesian Classifier")
      buttons += new RadioButton("SVM")
    }
    //第一组参数
    var R1 = new ComboBox(List(5,10,20,30,40,50))
    var R2 = new ComboBox(List(5,10,15,20,25,30))
    var R3 = new ComboBox(List(0,5,10,20,30,40,50))
    //第二组参数
    var N1 = new ComboBox(List(0,1,5,10))
    //第三组参数
    var L1 = new ComboBox(List(20,50,100,150))
    var L2 = new ComboBox(List(0.0,0.1,0.5,1.0,2.0))
    var L3 = new ComboBox(List(true,false))
    var L4 = new ComboBox(List(true,false))
    var button = new Button { text = "Start Prediction" }
    var label = new Label { text = "there is no result"}
    var label2 = new Label{icon = new ImageIcon("src/main/scala/edu/neu/coe/csye7200/image/monkey.jpg")}
    //页面布局
    contents = new BoxPanel(Orientation.Vertical) {
      //input section
      contents += new GridPanel(1, 1) {
        contents += new BorderPanel {
          layout += new Label("Input tweet:") -> West
          layout += textField -> Center
        }
      }
      contents += new Separator
      //choose model section & Set section
      contents += new GridPanel(4, 1) {
        //choose model
        contents += new BorderPanel {
          layout += new Label("Choose Model:") -> West
          layout += new FlowPanel {
            contents ++= buttonGroup.buttons
          } -> Center
        }
        //set1
        contents += new BorderPanel {
          layout += new Label("Random Forest Classifier:") -> West
          layout += new FlowPanel {
            contents += new Label("Num of trees:")
            contents += R1
            contents += new Label("Maxdepth of trees:")
            contents += R2
            contents += new Label("Random Seed:")
            contents += R3
          } -> Center
        }
        //set2
        contents += new BorderPanel {
          layout += new Label("Naive Bayesian Classifier:") -> West
          layout += new FlowPanel {
            contents += new Label("Smoothing:")
            contents += N1
          } -> Center
        }
        //set3
        contents += new BorderPanel {
          layout += new Label("SVM:") -> West
          layout += new FlowPanel {
            contents += new Label("Max iterations:")
            contents += L1
            contents += new Label("Regularization parameter:")
            contents += L2
            contents += new Label("Standardize the training features")
            contents += L3
            contents += new Label("Fit an intercept term:")
            contents += L4
          } -> Center
        }
      }
      contents += new Separator
      //button & result
      contents += button
      contents += label2
      contents += label
      border = Swing.EmptyBorder(30, 30, 30, 30)
    }

    size = new Dimension(800,800)
    var clicks = 0
    listenTo(button)
    reactions += {
      case ButtonClicked(b) if b == button =>
        //下面是点击button之后要做的操作
        //获取文本信息用textField.text
        //结果展示放到lavel.text = 这里
        clicks += 1
      //获取选中的radio的内容
      //label.text =

      val text = textField.text
      val num_of_tree = R1.selection.item
      val max_depth = R2.selection.item
      val seed = R3.selection.item
      val smoothing = N1.selection.item
      val max_iter = L1.selection.item
      val regularization = L2.selection.item
      val standardize = L3.selection.item
      val fit = L4.selection.item
      val list = List(text ,num_of_tree ,max_depth, seed, smoothing, max_iter, regularization, regularization, standardize, fit)
      println(list)
    }

  }
}
